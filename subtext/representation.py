"""Low-capacity candidate-direction fitting and held-out evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .identities import sha256_file
from .pilot_analysis import GROUPS, factorial_contrast


def residualize_nuisance(
    torch_module: Any,
    development_activations: Any,
    development_nuisances: Any,
    evaluation_activations: Any,
    evaluation_nuisances: Any,
    *,
    ridge: float = 1e-3,
) -> dict[str, Any]:
    """Fit nuisance regression on development only and apply it out of sample."""

    if ridge <= 0:
        raise ValueError("ridge must be positive")
    x_dev = development_activations.float()
    z_dev = development_nuisances.float()
    x_eval = evaluation_activations.float()
    z_eval = evaluation_nuisances.float()
    if x_dev.ndim != 2 or z_dev.ndim != 2 or x_eval.ndim != 2 or z_eval.ndim != 2:
        raise ValueError("activations and nuisance matrices must be two-dimensional")
    if x_dev.shape[0] != z_dev.shape[0] or x_eval.shape[0] != z_eval.shape[0]:
        raise ValueError("activation and nuisance row counts must match")
    if z_dev.shape[1] != z_eval.shape[1] or x_dev.shape[1] != x_eval.shape[1]:
        raise ValueError("development/evaluation feature dimensions must match")
    x_mean = x_dev.mean(dim=0)
    z_mean = z_dev.mean(dim=0)
    centered_z = z_dev - z_mean
    centered_x = x_dev - x_mean
    identity = torch_module.eye(z_dev.shape[1], dtype=centered_z.dtype, device=centered_z.device)
    coefficients = torch_module.linalg.solve(
        centered_z.transpose(0, 1) @ centered_z + ridge * identity,
        centered_z.transpose(0, 1) @ centered_x,
    )
    return {
        "development_residuals": centered_x - centered_z @ coefficients,
        "evaluation_residuals": (x_eval - x_mean) - (z_eval - z_mean) @ coefficients,
        "coefficients": coefficients,
        "activation_mean": x_mean,
        "nuisance_mean": z_mean,
    }


def fit_interaction_direction(
    torch_module: Any, grouped_development_vectors: Mapping[tuple[str, str], Any]
) -> dict[str, Any]:
    """Fit the one-dimensional who × outcome direction on development data."""

    contrast = factorial_contrast(torch_module, grouped_development_vectors)
    vector = contrast["interaction"].float()
    norm = vector.norm()
    if float(norm.item()) <= 1e-12:
        raise ValueError("development interaction direction has zero norm")
    return {"basis": (vector / norm).reshape(-1, 1), "contrast": contrast}


def family_interactions(
    torch_module: Any,
    basis: Any,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Score a frozen direction and compute the factorial interaction by family."""

    if basis.ndim != 2 or basis.shape[1] != 1:
        raise ValueError("candidate basis must have shape [d_model, 1]")
    families = sorted({str(row["task_family"]) for row in rows})
    output: dict[str, float] = {}
    for family in families:
        grouped: dict[tuple[str, str], list[float]] = {group: [] for group in GROUPS}
        for row in rows:
            if row["task_family"] != family:
                continue
            vector = row["activation"].float()
            score = float((vector @ basis[:, 0]).item())
            grouped[(str(row["who"]), str(row["outcome"]))].append(score)
        missing = [group for group, values in grouped.items() if not values]
        if missing:
            raise ValueError(f"family {family!r} is missing groups: {missing}")
        means = {group: sum(values) / len(values) for group, values in grouped.items()}
        output[family] = (
            means[("self", "blocked")]
            - means[("self", "satisfied")]
            - means[("other", "blocked")]
            + means[("other", "satisfied")]
        )
    return output


def representation_gate(
    heldout_interactions: Mapping[str, float],
    nuisance_correlations: Mapping[str, float],
    *,
    minimum_heldout_families: int,
    maximum_absolute_nuisance_correlation: float,
) -> dict[str, Any]:
    """Require held-out sign generalization and separation from every nuisance."""

    heldout_pass = (
        len(heldout_interactions) >= minimum_heldout_families
        and all(value > 0 for value in heldout_interactions.values())
    )
    nuisance_pass = bool(nuisance_correlations) and all(
        abs(value) <= maximum_absolute_nuisance_correlation
        for value in nuisance_correlations.values()
    )
    return {
        "passed": heldout_pass and nuisance_pass,
        "heldout_generalization_passed": heldout_pass,
        "nuisance_controls_passed": nuisance_pass,
        "heldout_family_count": len(heldout_interactions),
        "heldout_interactions": dict(heldout_interactions),
        "nuisance_correlations": dict(nuisance_correlations),
    }


def freeze_candidate_artifact(
    path: str | Path,
    torch_module: Any,
    basis: Any,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a frozen basis plus hash-bearing metadata, refusing overwrite."""

    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=False)
    basis_path = destination / "basis.pt"
    torch_module.save(basis.detach().cpu(), basis_path)
    payload = {
        **dict(metadata),
        "frozen": True,
        "basis_file": basis_path.name,
        "basis_sha256": sha256_file(basis_path),
    }
    (destination / "candidate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
