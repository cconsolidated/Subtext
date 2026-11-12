"""Fail-closed residual-subspace interventions and matched shams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class InterventionGateError(RuntimeError):
    """Raised when causal edits are requested before observational gates pass."""


def ensure_intervention_ready(metadata: Mapping[str, Any]) -> None:
    required = {
        "frozen": True,
        "gate_status": "passed",
        "heldout_generalization_passed": True,
        "nuisance_controls_passed": True,
    }
    failures = [key for key, expected in required.items() if metadata.get(key) != expected]
    if int(metadata.get("heldout_family_count", 0)) < 2:
        failures.append("heldout_family_count")
    if failures:
        raise InterventionGateError(
            "candidate artifact is not authorized for intervention: "
            + ", ".join(sorted(set(failures)))
        )


def validate_basis(torch_module: Any, basis: Any, *, tolerance: float = 1e-5) -> None:
    if basis.ndim != 2 or basis.shape[1] < 1:
        raise ValueError("basis must have shape [d_model, rank]")
    gram = basis.float().transpose(0, 1) @ basis.float()
    identity = torch_module.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    if not bool(torch_module.allclose(gram, identity, atol=tolerance, rtol=tolerance)):
        raise ValueError("basis columns must be orthonormal")


def suppress_subspace(torch_module: Any, vector: Any, basis: Any, dose: float) -> Any:
    validate_basis(torch_module, basis)
    if not 0 <= dose <= 1:
        raise ValueError("suppression dose must be in [0, 1]")
    projection = basis @ (basis.transpose(0, 1) @ vector)
    return vector - dose * projection


def source_patch(torch_module: Any, target: Any, source: Any, basis: Any, dose: float) -> Any:
    validate_basis(torch_module, basis)
    if target.shape != source.shape:
        raise ValueError("source and target vectors must have equal shapes")
    if not 0 <= dose <= 1:
        raise ValueError("patch dose must be in [0, 1]")
    delta = basis @ (basis.transpose(0, 1) @ (source - target))
    return target + dose * delta


def random_subspace_like(torch_module: Any, basis: Any, seed: int) -> Any:
    """Construct a seeded orthonormal random sham with equal dimensionality."""

    validate_basis(torch_module, basis)
    generator = torch_module.Generator(device="cpu")
    generator.manual_seed(seed)
    random = torch_module.randn(
        basis.shape,
        generator=generator,
        dtype=torch_module.float32,
        device="cpu",
    ).to(basis.device)
    q, _r = torch_module.linalg.qr(random, mode="reduced")
    return q.to(dtype=basis.dtype)


def scrambled_subspace(torch_module: Any, basis: Any, seed: int) -> Any:
    """Create a deterministic coordinate-scrambled candidate sham."""

    validate_basis(torch_module, basis)
    generator = torch_module.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch_module.randperm(basis.shape[0], generator=generator)
    signs = torch_module.where(
        torch_module.rand(basis.shape[0], generator=generator) >= 0.5,
        1.0,
        -1.0,
    ).to(dtype=basis.dtype)
    scrambled = basis.detach().cpu()[permutation] * signs[:, None]
    q, _r = torch_module.linalg.qr(scrambled.float(), mode="reduced")
    return q.to(device=basis.device, dtype=basis.dtype)


def match_delta_norm(torch_module: Any, original: Any, edited: Any, target_norm: float) -> Any:
    if target_norm < 0:
        raise ValueError("target edit norm must be non-negative")
    delta = edited - original
    current = float(delta.float().norm().item())
    if current <= 1e-12:
        if target_norm <= 1e-12:
            return original.clone()
        raise ValueError("cannot norm-match a zero edit")
    return original + delta * (target_norm / current)


def relative_norm_mismatch(target_norms: Sequence[float], sham_norms: Sequence[float]) -> float:
    if not target_norms or len(target_norms) != len(sham_norms):
        raise ValueError("target and sham norm lists must be non-empty and matched")
    target_mean = sum(target_norms) / len(target_norms)
    sham_mean = sum(sham_norms) / len(sham_norms)
    if target_mean <= 1e-12:
        return 0.0 if sham_mean <= 1e-12 else float("inf")
    return abs(sham_mean - target_mean) / target_mean


@dataclass(frozen=True)
class ResidualSubspaceIntervention:
    """Hook-compatible edit at frozen layers and named token positions."""

    identity: str
    torch_module: Any
    operation: str
    basis: Any
    dose: float
    target_layers: tuple[int, ...]
    target_position_names: tuple[str, ...]
    candidate_metadata: Mapping[str, Any]
    source_vectors: Mapping[tuple[int, str], Any] | None = None

    def __post_init__(self) -> None:
        ensure_intervention_ready(self.candidate_metadata)
        validate_basis(self.torch_module, self.basis)
        if self.operation not in {"suppression", "source_patch"}:
            raise ValueError("unsupported intervention operation")
        if not 0 <= self.dose <= 1:
            raise ValueError("intervention dose must be in [0, 1]")
        if self.operation == "source_patch" and self.source_vectors is None:
            raise ValueError("source_patch requires source_vectors")

    def edit(
        self,
        hidden: Any,
        *,
        layer: int,
        positions: Mapping[str, int],
    ) -> Any:
        if layer not in self.target_layers:
            return hidden
        edited_hidden = hidden.clone()
        for name in self.target_position_names:
            if name not in positions:
                raise KeyError(f"target position {name!r} was not resolved")
            position = positions[name]
            vector = edited_hidden[0, position]
            if self.operation == "suppression":
                edited = suppress_subspace(
                    self.torch_module, vector, self.basis, self.dose
                )
            else:
                assert self.source_vectors is not None
                source = self.source_vectors[(layer, name)].to(
                    device=vector.device, dtype=vector.dtype
                )
                edited = source_patch(
                    self.torch_module, vector, source, self.basis, self.dose
                )
            edited_hidden[0, position] = edited
        return edited_hidden
