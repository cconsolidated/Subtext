"""Subtext live Jacobian-lens viewer.

Importing this module creates the FastAPI application but does not load a model
or touch an accelerator. Run it as a script (or call ``initialize_runtime``)
to load the configured artifacts explicitly.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import pathlib
import re
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from subtext.config import SubtextConfig, load_config
from subtext.models import ModelBundle, load_model_bundle


HERE = pathlib.Path(__file__).parent
DEFAULT_CONFIG = HERE / "config" / "server.default.toml"


@dataclass
class RuntimeState:
    config: SubtextConfig
    bundle: ModelBundle
    viz_layers: tuple[int, ...]
    layer_depth: dict[int, float]
    display_mask: Any


_runtime: RuntimeState | None = None


def _resolve_viz_layers(config: SubtextConfig, bundle: ModelBundle) -> tuple[int, ...]:
    n_layers = bundle.model_adapter.n_layers
    wanted = [round(fraction * (n_layers - 1)) for fraction in config.viewer.layer_fractions]
    return tuple(
        sorted(
            {
                min(bundle.lens.source_layers, key=lambda source, target=target: abs(source - target))
                for target in wanted
            }
        )
    )


def _build_display_mask(config: SubtextConfig, bundle: ModelBundle) -> Any:
    torch = bundle.torch
    tokenizer = bundle.tokenizer
    mask_path = HERE / config.viewer.token_mask_file
    vocab_size = bundle.hf_model.get_output_embeddings().weight.shape[0]
    if mask_path.exists():
        return torch.load(mask_path, weights_only=True).to(bundle.device)

    print("[subtext] building word-like token mask (one-time, ~1 min) ...")
    display_mask = bundle.meaningful_token_mask(
        tokenizer, vocab_size, torch.device("cpu")
    )
    for token_id in display_mask.nonzero().flatten().tolist():
        raw = tokenizer.decode([token_id])
        stripped = raw.strip()
        if not (
            raw.startswith(" ")
            and stripped.isascii()
            and re.match(r"^[A-Za-z][A-Za-z'\-]+$", stripped)
            and len(stripped) > 2
        ):
            display_mask[token_id] = False
    torch.save(display_mask, mask_path)
    return display_mask.to(bundle.device)


def initialize_runtime(config_path: str | pathlib.Path = DEFAULT_CONFIG) -> RuntimeState:
    """Load configured artifacts and prepare the viewer runtime exactly once."""

    global _runtime
    if _runtime is not None:
        return _runtime
    config = load_config(config_path)
    print(f"[subtext] loading {config.artifacts.model.repository} ...")
    bundle = load_model_bundle(config)
    viz_layers = _resolve_viz_layers(config, bundle)
    n_layers = bundle.model_adapter.n_layers
    layer_depth = {layer: layer / (n_layers - 1) for layer in viz_layers}
    bundle.lens.jacobians = {
        layer: bundle.lens.jacobians[layer].to(bundle.device) for layer in viz_layers
    }
    display_mask = _build_display_mask(config, bundle)
    _runtime = RuntimeState(
        config=config,
        bundle=bundle,
        viz_layers=viz_layers,
        layer_depth=layer_depth,
        display_mask=display_mask,
    )
    print(f"[subtext] reading layers {list(viz_layers)} of {n_layers}")
    print(f"[subtext] display vocabulary: {int(display_mask.sum())} word tokens")
    return _runtime


def require_runtime() -> RuntimeState:
    if _runtime is None:
        raise RuntimeError(
            "Subtext runtime is not initialized; call initialize_runtime() before serving chats"
        )
    return _runtime


class ResidualCatcher:
    """Keep each selected decoder block's most recent hidden-state output."""

    def __init__(self, layers: Any, indices: tuple[int, ...]):
        self.acts: dict[int, Any] = {}
        self._handles = [
            layers[index].register_forward_hook(self._make(index)) for index in indices
        ]

    def _make(self, index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            self.acts[index] = hidden.detach()

        return hook

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


def readout(residuals: dict[int, Any], position: int) -> list[dict[str, Any]]:
    """Aggregate lens top-k results across configured visualization layers."""

    runtime = require_runtime()
    bundle = runtime.bundle
    words: dict[str, dict[str, Any]] = {}
    for layer in runtime.viz_layers:
        hidden = residuals[layer][0, position].float()
        transported = bundle.lens.transport(hidden, layer)
        logits = bundle.model_adapter.unembed(transported).float()
        probabilities = bundle.torch.softmax(logits, -1)
        probabilities = probabilities.masked_fill(~runtime.display_mask, 0.0)
        top = probabilities.topk(runtime.config.viewer.top_k_per_layer)
        for probability, token_id in zip(top.values.tolist(), top.indices.tolist()):
            if probability <= 0.0:
                continue
            word = bundle.tokenizer.decode([token_id]).strip().lower()
            entry = words.setdefault(
                word,
                {"p": 0.0, "pd": 0.0, "psum": 0.0, "prof": {}},
            )
            entry["p"] = max(entry["p"], probability)
            entry["pd"] += probability * runtime.layer_depth[layer]
            entry["psum"] += probability
            entry["prof"][layer] = max(
                entry["prof"].get(layer, 0.0), probability
            )
    output = [
        {
            "w": word,
            "p": round(entry["p"], 5),
            "d": round(entry["pd"] / entry["psum"], 3),
            "prof": [
                [layer, round(probability, 5)]
                for layer, probability in sorted(entry["prof"].items())
            ],
        }
        for word, entry in words.items()
    ]
    output.sort(key=lambda item: -item["p"])
    return output[: runtime.config.viewer.max_words_per_frame]


def _template_ids(messages: list[dict[str, Any]], add_generation: bool) -> Any:
    runtime = require_runtime()
    output = runtime.bundle.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation,
        return_tensors="pt",
        enable_thinking=False,
    )
    return output if isinstance(output, runtime.bundle.torch.Tensor) else output["input_ids"]


def encode_chat(messages: list[dict[str, Any]]) -> tuple[Any, int]:
    """Tokenize a conversation and locate the newest user-message boundary."""

    runtime = require_runtime()
    torch = runtime.bundle.torch
    input_ids = _template_ids(messages, True).to(runtime.bundle.model_adapter.input_device)
    previous = (
        _template_ids(messages[:-1], False)
        if len(messages) > 1
        else torch.zeros(1, 0, dtype=torch.long)
    )
    return input_ids, min(previous.shape[1], input_ids.shape[1] - 1)


app = FastAPI()


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.websocket("/ws")
async def ws_chat(websocket: WebSocket):
    runtime = require_runtime()
    bundle = runtime.bundle
    torch = bundle.torch
    await websocket.accept()
    await websocket.send_text(
        json.dumps(
            {
                "type": "hello",
                "model": runtime.config.artifacts.model.repository.split("/")[-1],
                "n_layers": bundle.model_adapter.n_layers,
                "layers": runtime.viz_layers,
            }
        )
    )
    try:
        while True:
            request = json.loads(await websocket.receive_text())
            if request.get("type") != "chat":
                continue
            messages = request["messages"]
            max_new = min(int(request.get("max_tokens", 220)), 512)
            temperature = float(request.get("temperature", 0.8))

            input_ids, read_from = encode_chat(messages)
            catcher = ResidualCatcher(bundle.model_adapter.layers, runtime.viz_layers)
            try:
                with torch.no_grad():
                    model_output = bundle.hf_model(input_ids=input_ids, use_cache=True)
                cache = model_output.past_key_values
                residuals = dict(catcher.acts)
                for position in range(read_from, input_ids.shape[1]):
                    token_text = bundle.tokenizer.decode([input_ids[0, position].item()])
                    frame = {
                        "type": "frame",
                        "phase": "reading",
                        "tok": token_text,
                        "thoughts": readout(residuals, position),
                    }
                    await websocket.send_text(json.dumps(frame))
                    await asyncio.sleep(0)

                eos_ids = bundle.hf_model.generation_config.eos_token_id
                eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
                logits = model_output.logits[:, -1]
                reply_ids: list[int] = []
                for _ in range(max_new):
                    probabilities = torch.softmax(logits.float() / temperature, -1)
                    sorted_probs, sorted_indices = probabilities.sort(descending=True)
                    keep = sorted_probs.cumsum(-1) - sorted_probs < 0.95
                    sorted_probs = sorted_probs * keep
                    next_token = sorted_indices[
                        0, torch.multinomial(sorted_probs[0] / sorted_probs.sum(), 1)
                    ].item()
                    if next_token in eos_ids:
                        break
                    reply_ids.append(next_token)
                    with torch.no_grad():
                        model_output = bundle.hf_model(
                            input_ids=torch.tensor(
                                [[next_token]], device=bundle.model_adapter.input_device
                            ),
                            past_key_values=cache,
                            use_cache=True,
                        )
                    cache = model_output.past_key_values
                    logits = model_output.logits[:, -1]
                    frame = {
                        "type": "frame",
                        "phase": "thinking",
                        "out": bundle.tokenizer.decode([next_token]),
                        "thoughts": readout(catcher.acts, -1),
                    }
                    await websocket.send_text(json.dumps(frame))
                    await asyncio.sleep(0)

                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "done",
                            "text": bundle.tokenizer.decode(
                                reply_ids, skip_special_tokens=True
                            ),
                        }
                    )
                )
            finally:
                catcher.close()
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Subtext local viewer")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=DEFAULT_CONFIG,
        help=f"validated TOML/JSON/YAML configuration (default: {DEFAULT_CONFIG})",
    )
    arguments = parser.parse_args()
    state = initialize_runtime(arguments.config)
    print(f"[subtext] ready -> http://localhost:{state.config.viewer.port}")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=state.config.viewer.port,
        log_level="warning",
    )
