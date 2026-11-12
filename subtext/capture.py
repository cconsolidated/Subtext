"""Framework-neutral selected-position activation capture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


class CaptureError(RuntimeError):
    """Raised when hook points, layers, or positions cannot be captured safely."""


class HookHandle(Protocol):
    def remove(self) -> None: ...


class ModelAdapter(Protocol):
    """Minimal adapter contract required by :class:`ActivationCapture`."""

    def get_hook_module(self, layer: int, hook_point: str) -> Any: ...

    def forward(self, input_ids: Any) -> Any: ...


@dataclass(frozen=True)
class CapturedForward:
    output: Any
    activations: Mapping[str, Mapping[int, Any]]
    pre_activations: Mapping[str, Mapping[int, Any]] | None = None


def _detach_clone_cpu(tensor: Any) -> Any:
    for operation in ("detach", "clone", "cpu"):
        method = getattr(tensor, operation, None)
        if method is None:
            raise CaptureError(f"captured value does not support {operation}()")
        tensor = method()
    return tensor


class ActivationCapture:
    """Capture named token positions at explicit residual-stream hook points.

    The adapter makes this usable with Hugging Face models as well as tiny fake
    models. Captured vectors are detached, cloned, and moved to CPU inside the
    hook so no live computation graph or device tensor leaks into persistence.
    """

    def __init__(
        self,
        model_adapter: ModelAdapter,
        layers: Sequence[int],
        hook_point: str,
    ) -> None:
        normalized_layers = tuple(layers)
        if not normalized_layers or any(
            not isinstance(layer, int) or isinstance(layer, bool) or layer < 0
            for layer in normalized_layers
        ):
            raise ValueError("layers must contain non-negative integers")
        if len(set(normalized_layers)) != len(normalized_layers):
            raise ValueError("layers must not contain duplicates")
        if not hook_point:
            raise ValueError("hook_point must not be empty")
        self.model_adapter = model_adapter
        self.layers = normalized_layers
        self.hook_point = hook_point

    def run(
        self,
        input_ids: Any,
        capture_positions: Mapping[str, int],
        intervention: Any | None = None,
    ) -> CapturedForward:
        if not capture_positions:
            raise CaptureError("at least one named capture position is required")
        for name, position in capture_positions.items():
            if not isinstance(name, str) or not name:
                raise CaptureError("capture position names must be non-empty strings")
            if (
                not isinstance(position, int)
                or isinstance(position, bool)
                or position < 0
            ):
                raise CaptureError(
                    f"capture position {name!r} must be a non-negative integer"
                )

        activations: dict[str, dict[int, Any]] = {
            name: {} for name in capture_positions
        }
        pre_activations: dict[str, dict[int, Any]] | None = (
            {name: {} for name in capture_positions} if intervention is not None else None
        )
        handles: list[HookHandle] = []

        def make_hook(layer: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                if pre_activations is not None:
                    for name, position in capture_positions.items():
                        pre_activations[name][layer] = _detach_clone_cpu(hidden[0, position])
                    hidden = intervention.edit(
                        hidden,
                        layer=layer,
                        positions=capture_positions,
                    )
                for name, position in capture_positions.items():
                    try:
                        vector = hidden[0, position]
                    except (IndexError, TypeError) as exc:
                        raise CaptureError(
                            f"position {name!r}={position} is invalid at layer {layer}"
                        ) from exc
                    activations[name][layer] = _detach_clone_cpu(vector)
                if intervention is not None:
                    if isinstance(output, tuple):
                        return (hidden, *output[1:])
                    return hidden
                return None

            return hook

        try:
            for layer in self.layers:
                module = self.model_adapter.get_hook_module(layer, self.hook_point)
                handles.append(module.register_forward_hook(make_hook(layer)))
            output = self.model_adapter.forward(input_ids)
        finally:
            for handle in handles:
                handle.remove()

        missing = {
            name: sorted(set(self.layers) - set(layer_map))
            for name, layer_map in activations.items()
            if set(layer_map) != set(self.layers)
        }
        if missing:
            raise CaptureError(f"hooks did not fire for all requested layers: {missing}")
        return CapturedForward(
            output=output,
            activations=activations,
            pre_activations=pre_activations,
        )
