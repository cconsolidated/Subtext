"""Explicit model/tokenizer/lens loading for the viewer and research runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import SubtextConfig


@dataclass(frozen=True)
class ModelBundle:
    torch: Any
    hf_model: Any
    tokenizer: Any
    model_adapter: Any
    lens: Any
    meaningful_token_mask: Any
    device: str
    dtype: Any


@dataclass(frozen=True)
class HuggingFaceResidualAdapter:
    """Expose the capture protocol for a loaded Hugging Face/JLens bundle."""

    bundle: ModelBundle

    def get_hook_module(self, layer: int, hook_point: str) -> Any:
        if hook_point != "residual_stream_block_output":
            raise KeyError(f"unsupported hook point: {hook_point}")
        return self.bundle.model_adapter.layers[layer]

    def forward(self, input_ids: Any) -> Any:
        return self.bundle.hf_model(input_ids=input_ids, use_cache=False)


def choose_device(torch_module: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype(torch_module: Any, requested: str, device: str) -> Any:
    if requested == "auto":
        return torch_module.bfloat16 if device != "cpu" else torch_module.float32
    return {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }[requested]


def _revision_kwargs(revision: str) -> dict[str, str]:
    return {"revision": revision} if revision else {}


def load_model_bundle(config: SubtextConfig) -> ModelBundle:
    """Load heavyweight dependencies and artifacts only when explicitly called."""

    import torch
    import transformers
    import jlens
    from jlens.vis import _meaningful_token_mask

    device = choose_device(torch, config.runtime.device)
    dtype = choose_dtype(torch, config.runtime.dtype, device)
    if config.runtime.deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    model_ref = config.artifacts.model
    tokenizer_ref = config.artifacts.tokenizer
    lens_ref = config.artifacts.lens
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_ref.repository,
        dtype=dtype,
        **_revision_kwargs(model_ref.revision),
    ).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_ref.repository,
        **_revision_kwargs(tokenizer_ref.revision),
    )
    model_adapter = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(
        lens_ref.repository,
        filename=lens_ref.file,
        **_revision_kwargs(lens_ref.revision),
    )
    return ModelBundle(
        torch=torch,
        hf_model=hf_model,
        tokenizer=tokenizer,
        model_adapter=model_adapter,
        lens=lens,
        meaningful_token_mask=_meaningful_token_mask,
        device=device,
        dtype=dtype,
    )
