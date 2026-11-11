"""Compare Subtext's live hook path with ``JacobianLens.apply``.

Run with the server stopped because both paths need the configured accelerator.
The configured model, tokenizer, and lens revisions are used verbatim.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from subtext.config import load_config
from subtext.models import load_model_bundle


HERE = Path(__file__).parent
DEFAULT_CONFIG = HERE / "config" / "server.default.toml"
PROMPT = "Fact: The currency used in the country shaped like a boot is"
LAYERS = [8, 14, 20, 26]
POSITIONS = [-4, -2, -1]
TOP_K = 5


class Catcher:
    def __init__(self, layers: Any, indices: list[int]):
        self.activations: dict[int, Any] = {}
        self.handles = [
            layers[index].register_forward_hook(self._make_hook(index))
            for index in indices
        ]

    def _make_hook(self, index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            self.activations[index] = hidden.detach()

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def main(config_path: Path) -> int:
    config = load_config(config_path)
    print("loading configured model + lens ...")
    bundle = load_model_bundle(config)
    torch = bundle.torch

    reference_logits, _, input_ids = bundle.lens.apply(
        bundle.model_adapter,
        PROMPT,
        layers=LAYERS,
        positions=POSITIONS,
    )

    catcher = Catcher(bundle.model_adapter.layers, LAYERS)
    try:
        with torch.no_grad():
            bundle.hf_model(input_ids=input_ids, use_cache=True)
    finally:
        catcher.close()

    print(f"prompt tokens: {input_ids.shape[1]}\n")
    all_ok = True
    for layer in LAYERS:
        for position_index, position in enumerate(POSITIONS):
            hidden = catcher.activations[layer][0, position].float()
            live_logits = bundle.model_adapter.unembed(
                bundle.lens.transport(hidden, layer)
            ).float().cpu()
            expected_logits = reference_logits[layer][position_index]
            live_top = live_logits.topk(TOP_K).indices.tolist()
            expected_top = expected_logits.topk(TOP_K).indices.tolist()
            matches = live_top == expected_top
            similarity = torch.nn.functional.cosine_similarity(
                live_logits, expected_logits, dim=0
            ).item()
            all_ok &= matches
            words = [bundle.tokenizer.decode([token]).strip() for token in expected_top]
            label = "MATCH" if matches else "MISMATCH"
            print(
                f"L{layer:>2} pos{position:>3} top{TOP_K} {label} "
                f"cos={similarity:.6f} ref={words}"
            )
            if not matches:
                live_words = [
                    bundle.tokenizer.decode([token]).strip() for token in live_top
                ]
                print(f"      live={live_words}")

    result = "PASS: live path == reference" if all_ok else "FAIL: paths differ"
    print(f"\n{result}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    raise SystemExit(main(arguments.config))
