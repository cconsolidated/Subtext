"""Shared-space analysis and self-contained reporting for pilot runs."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


GROUPS = (
    ("self", "satisfied"),
    ("self", "blocked"),
    ("other", "satisfied"),
    ("other", "blocked"),
)


def _rms(torch_module: Any, vector: Any) -> float:
    return float(vector.float().square().mean().sqrt().item())


def _cosine(torch_module: Any, first: Any, second: Any) -> float:
    denominator = first.float().norm() * second.float().norm()
    if float(denominator.item()) <= 1e-12:
        return 0.0
    return float(torch_module.dot(first.float(), second.float()).div(denominator).item())


def factorial_contrast(torch_module: Any, grouped: Mapping[tuple[str, str], Any]) -> dict[str, Any]:
    """Compute blocked effects and their who × outcome interaction."""

    missing = [group for group in GROUPS if group not in grouped]
    if missing:
        raise ValueError(f"missing factorial groups: {missing}")
    self_effect = grouped[("self", "blocked")].float() - grouped[("self", "satisfied")].float()
    other_effect = grouped[("other", "blocked")].float() - grouped[("other", "satisfied")].float()
    interaction = self_effect - other_effect
    return {
        "self_effect": self_effect,
        "other_effect": other_effect,
        "interaction": interaction,
        "self_effect_rms": _rms(torch_module, self_effect),
        "other_effect_rms": _rms(torch_module, other_effect),
        "interaction_rms": _rms(torch_module, interaction),
        "effect_cosine": _cosine(torch_module, self_effect, other_effect),
    }


def shared_pca(torch_module: Any, matrix: Any, dimensions: int = 2) -> dict[str, Any]:
    """Fit one PCA basis to all rows and return shared coordinates."""

    values = matrix.float()
    if values.ndim != 2 or values.shape[0] < dimensions + 1:
        raise ValueError("PCA requires a 2D matrix with more rows than dimensions")
    center = values.mean(dim=0)
    centered = values - center
    _u, singular, vh = torch_module.linalg.svd(centered, full_matrices=False)
    basis = vh[:dimensions].transpose(0, 1).contiguous()
    coordinates = centered @ basis
    variance = singular.square()
    total = variance.sum().clamp_min(1e-12)
    explained = variance[:dimensions] / total
    return {
        "center": center,
        "basis": basis,
        "coordinates": coordinates,
        "explained_variance_ratio": [float(value.item()) for value in explained],
    }


def _mean_vectors(torch_module: Any, rows: Sequence[Any]) -> Any:
    if not rows:
        raise ValueError("cannot average an empty vector group")
    return torch_module.stack([row.float() for row in rows]).mean(dim=0)


def _load_run(run_dir: Path, torch_module: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episode_dir = run_dir / "episodes"
    episodes = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(episode_dir.glob("*.json"))
    ]
    if not episodes:
        raise ValueError(f"no completed episodes found in {episode_dir}")
    tensors: dict[str, Any] = {}
    for episode in episodes:
        tensor_path = run_dir / episode["tensor_file"]
        tensors[episode["episode_id"]] = torch_module.load(
            tensor_path, map_location="cpu", weights_only=True
        )
    return episodes, tensors


def select_analysis_episodes(
    episodes: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], str]:
    """Apply the calibration gate, retaining a fallback for gate-failure diagnostics."""

    admitted = [episode for episode in episodes if bool(episode["preference_admitted"])]
    if admitted:
        return admitted, "calibration_admitted_episodes"
    return list(episodes), "all_instrumentation_episodes_no_admitted_data"


def _layer_contrast_rows(
    torch_module: Any,
    episodes: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Any],
    positions: Sequence[str],
    layers: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    contrasts: dict[tuple[str, int], dict[str, Any]] = {}
    for position in positions:
        for layer in layers:
            grouped_rows: dict[tuple[str, str], list[Any]] = {group: [] for group in GROUPS}
            for episode in episodes:
                grouped_rows[(episode["who"], episode["outcome"])].append(
                    tensors[episode["episode_id"]][position][layer]
                )
            means = {
                group: _mean_vectors(torch_module, vectors)
                for group, vectors in grouped_rows.items()
            }
            contrast = factorial_contrast(torch_module, means)
            contrasts[(position, layer)] = contrast
            rows.append(
                {
                    "position": position,
                    "layer": layer,
                    "self_effect_rms": contrast["self_effect_rms"],
                    "other_effect_rms": contrast["other_effect_rms"],
                    "interaction_rms": contrast["interaction_rms"],
                    "effect_cosine": contrast["effect_cosine"],
                }
            )
    return rows, contrasts


def _family_direction_interactions(
    torch_module: Any,
    episodes: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Any],
    *,
    position: str,
    layer: int,
    direction: Any,
) -> dict[str, float]:
    output: dict[str, float] = {}
    families = sorted({str(episode["task_family"]) for episode in episodes})
    for family in families:
        grouped: dict[tuple[str, str], list[float]] = {group: [] for group in GROUPS}
        for episode in episodes:
            if episode["task_family"] != family:
                continue
            vector = tensors[episode["episode_id"]][position][layer].float()
            grouped[(episode["who"], episode["outcome"])].append(
                float(torch_module.dot(vector, direction).item())
            )
        if any(not values for values in grouped.values()):
            continue
        means = {group: sum(values) / len(values) for group, values in grouped.items()}
        output[family] = (
            means[("self", "blocked")]
            - means[("self", "satisfied")]
            - means[("other", "blocked")]
            + means[("other", "satisfied")]
        )
    return output


def analyze_run(run_dir: str | Path, torch_module: Any) -> dict[str, Any]:
    """Analyze exact saved activations in one shared cross-session space."""

    directory = Path(run_dir)
    episodes, tensors = _load_run(directory, torch_module)
    analysis_episodes, analysis_population = select_analysis_episodes(episodes)
    positions = sorted(next(iter(tensors.values())).keys())
    layers = sorted(int(layer) for layer in next(iter(tensors.values()))[positions[0]].keys())
    layer_rows, contrasts = _layer_contrast_rows(
        torch_module, analysis_episodes, tensors, positions, layers
    )
    development_episodes = [
        episode for episode in analysis_episodes if episode.get("split") == "development"
    ]
    if not development_episodes:
        development_episodes = list(analysis_episodes)
    development_rows, development_contrasts = _layer_contrast_rows(
        torch_module, development_episodes, tensors, positions, layers
    )

    decision_position = "pre_downstream_choice"
    candidate_rows = [
        row for row in development_rows if row["position"] == decision_position
    ]
    selected = max(candidate_rows, key=lambda row: row["interaction_rms"])
    selected_layer = int(selected["layer"])
    development_matrix = torch_module.stack(
        [
            tensors[episode["episode_id"]][decision_position][selected_layer].float()
            for episode in development_episodes
        ]
    )
    pca = shared_pca(torch_module, development_matrix)
    matrix = torch_module.stack(
        [
            tensors[episode["episode_id"]][decision_position][selected_layer].float()
            for episode in analysis_episodes
        ]
    )
    coordinates = (matrix - pca["center"]) @ pca["basis"]
    development_interaction = development_contrasts[
        (decision_position, selected_layer)
    ]["interaction"].float()
    direction_norm = development_interaction.norm().clamp_min(1e-12)
    development_direction = development_interaction / direction_norm
    interactions_by_split = {
        split: _family_direction_interactions(
            torch_module,
            [episode for episode in analysis_episodes if episode.get("split") == split],
            tensors,
            position=decision_position,
            layer=selected_layer,
            direction=development_direction,
        )
        for split in ("development", "validation", "test")
    }
    heldout_interactions = {
        **interactions_by_split["validation"],
        **interactions_by_split["test"],
    }
    heldout_sign_pass = bool(heldout_interactions) and all(
        value > 0 for value in heldout_interactions.values()
    )
    points = []
    for index, episode in enumerate(analysis_episodes):
        points.append(
            {
                "episode_id": episode["episode_id"],
                "scenario_id": episode["scenario_id"],
                "seed": episode["seed"],
                "who": episode["who"],
                "outcome": episode["outcome"],
                "split": episode.get("split", "development"),
                "x": float(coordinates[index, 0].item()),
                "y": float(coordinates[index, 1].item()),
                "primary_included": bool(episode["preference_admitted"]),
            }
        )
    group_coordinates: dict[str, dict[str, float]] = {}
    for who, outcome in GROUPS:
        matching = [
            coordinates[index]
            for index, episode in enumerate(analysis_episodes)
            if episode["who"] == who and episode["outcome"] == outcome
        ]
        mean = torch_module.stack(matching).mean(dim=0)
        group_coordinates[f"{who}/{outcome}"] = {
            "x": float(mean[0].item()),
            "y": float(mean[1].item()),
        }

    behavior_groups: dict[str, list[float]] = {f"{who}/{outcome}": [] for who, outcome in GROUPS}
    for episode in analysis_episodes:
        behavior_groups[f"{episode['who']}/{episode['outcome']}"].append(
            float(episode["sustain_log_odds"])
        )
    behavior_means = {
        group: sum(values) / len(values) for group, values in behavior_groups.items()
    }
    behavioral_interaction = (
        behavior_means["self/blocked"]
        - behavior_means["self/satisfied"]
        - behavior_means["other/blocked"]
        + behavior_means["other/satisfied"]
    )
    admitted = sum(bool(episode["preference_admitted"]) for episode in episodes)
    calibration_files = sorted((directory / "calibrations").glob("*.json"))
    calibrations = [
        json.loads(path.read_text(encoding="utf-8")) for path in calibration_files
    ]
    label_counts: dict[str, int] = {}
    for calibration in calibrations:
        for trial in calibration["trials"]:
            label = trial["chosen_label"]
            label_counts[label] = label_counts.get(label, 0) + 1
    calibration_trial_count = sum(label_counts.values())
    maximum_label_rate = (
        max(label_counts.values()) / calibration_trial_count
        if calibration_trial_count
        else 0.0
    )
    result = {
        "analysis_schema_version": "0.1",
        "episode_count": len(episodes),
        "analysis_episode_count": len(analysis_episodes),
        "primary_included_episode_count": admitted,
        "exploratory_only": True,
        "analysis_population": analysis_population,
        "calibrations": [
            {
                "scenario_id": item["scenario_id"],
                "preferred_activity_id": item["preferred_activity_id"],
                "consistent_choices": item["consistent_choices"],
                "trial_count": item["trial_count"],
                "reversal_stable": item["reversal_stable"],
                "preference_strength": item["preference_strength"],
                "admitted": item["admitted"],
            }
            for item in calibrations
        ],
        "calibration_answer_label_counts": label_counts,
        "calibration_maximum_label_choice_rate": maximum_label_rate,
        "behavior_group_mean_sustain_log_odds": behavior_means,
        "behavioral_who_outcome_interaction": behavioral_interaction,
        "selected_layer_exploratory": selected_layer,
        "layer_selection_population": "development_only",
        "development_family_count": len(
            {episode["task_family"] for episode in development_episodes}
        ),
        "direction_family_interactions_by_split": interactions_by_split,
        "representation_gate": {
            "passed": False,
            "heldout_sign_generalization_passed": heldout_sign_pass,
            "heldout_family_count": len(heldout_interactions),
            "nuisance_controls_passed": False,
            "reason": "nuisance-control acquisition and residualization are required",
        },
        "shared_pca_explained_variance_ratio": pca["explained_variance_ratio"],
        "group_coordinates": group_coordinates,
        "points": points,
        "layer_contrasts": layer_rows,
    }
    (directory / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (directory / "report.html").write_text(_render_html(result), encoding="utf-8")
    return result


def _color(value: float, maximum: float) -> str:
    ratio = 0.0 if maximum <= 0 else min(1.0, value / maximum)
    red = int(40 + 210 * ratio)
    green = int(55 + 70 * (1.0 - ratio))
    blue = int(90 + 80 * (1.0 - ratio))
    return f"rgb({red},{green},{blue})"


def _render_html(result: Mapping[str, Any]) -> str:
    points = result["points"]
    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)

    def sx(value: float) -> float:
        return 55 + 690 * (value - min_x) / span_x

    def sy(value: float) -> float:
        return 445 - 390 * (value - min_y) / span_y

    palette = {
        "self/satisfied": "#4ea1ff",
        "self/blocked": "#ff6b6b",
        "other/satisfied": "#77d49b",
        "other/blocked": "#d28cff",
    }
    circles = []
    for point in points:
        group = f"{point['who']}/{point['outcome']}"
        circles.append(
            f'<circle cx="{sx(point["x"]):.2f}" cy="{sy(point["y"]):.2f}" r="4.5" '
            f'fill="{palette[group]}" fill-opacity="0.7"><title>'
            f'{html.escape(point["episode_id"])} · {group} · '
            f'{html.escape(point["scenario_id"])}</title></circle>'
        )
    arrows = []
    groups = result["group_coordinates"]
    for who, color in (("self", "#ffffff"), ("other", "#f4c8ff")):
        start = groups[f"{who}/satisfied"]
        end = groups[f"{who}/blocked"]
        arrows.append(
            f'<line x1="{sx(start["x"]):.2f}" y1="{sy(start["y"]):.2f}" '
            f'x2="{sx(end["x"]):.2f}" y2="{sy(end["y"]):.2f}" '
            f'stroke="{color}" stroke-width="4" marker-end="url(#arrow)"><title>{who} blocked direction</title></line>'
        )
    decision_rows = [
        row for row in result["layer_contrasts"] if row["position"] == "pre_downstream_choice"
    ]
    maximum = max(row["interaction_rms"] for row in decision_rows)
    heat_cells = "".join(
        f'<div class="heat" style="background:{_color(row["interaction_rms"], maximum)}">'
        f'<b>L{row["layer"]}</b><span>{row["interaction_rms"]:.4f}</span></div>'
        for row in decision_rows
    )
    layer_table = "".join(
        "<tr>"
        f"<td>{html.escape(row['position'])}</td><td>{row['layer']}</td>"
        f"<td>{row['self_effect_rms']:.6f}</td><td>{row['other_effect_rms']:.6f}</td>"
        f"<td>{row['interaction_rms']:.6f}</td><td>{row['effect_cosine']:.3f}</td>"
        "</tr>"
        for row in result["layer_contrasts"]
    )
    behavior_rows = "".join(
        f"<tr><td>{html.escape(group)}</td><td>{value:.4f}</td></tr>"
        for group, value in result["behavior_group_mean_sustain_log_odds"].items()
    )
    variance = sum(result["shared_pca_explained_variance_ratio"]) * 100
    excluded = result["episode_count"] - result["primary_included_episode_count"]
    warning = (
        f"The primary charts use {result['analysis_episode_count']} calibration-admitted episodes; "
        f"{excluded} excluded episodes remain in the raw audit trail."
        if excluded
        else "All episodes passed the provisional calibration admission rule."
    )
    calibration_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['scenario_id'])}</td>"
        f"<td>{item['consistent_choices']}/{item['trial_count']}</td>"
        f"<td>{'yes' if item['reversal_stable'] else 'no'}</td>"
        f"<td>{'yes' if item['admitted'] else 'no'}</td>"
        "</tr>"
        for item in result["calibrations"]
    )
    label_counts = ", ".join(
        f"{html.escape(label)}={count}"
        for label, count in sorted(result["calibration_answer_label_counts"].items())
    )
    direction_rows = "".join(
        f"<tr><td>{html.escape(split)}</td><td>{html.escape(family)}</td><td>{value:.5f}</td></tr>"
        for split, families in result["direction_family_interactions_by_split"].items()
        for family, value in families.items()
    )
    representation_gate = result["representation_gate"]
    representation_status = "PASS" if representation_gate["passed"] else "HOLD"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Subtext M2 pilot</title>
<style>
body{{margin:0;background:#090b10;color:#e7eaf0;font:14px system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:34px}}
h1{{font-size:30px}}h2{{margin-top:34px}}.note{{padding:14px 16px;background:#151925;border-left:4px solid #ffbf69}}
.grid{{display:grid;grid-template-columns:1.4fr 1fr;gap:24px}}.card{{background:#10141d;border:1px solid #252b38;border-radius:12px;padding:18px}}
.heatmap{{display:grid;grid-template-columns:repeat(8,1fr);gap:6px}}.heat{{min-height:48px;padding:7px;border-radius:6px;display:flex;flex-direction:column;justify-content:space-between}}
.heat span{{font-size:11px}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #272d39;padding:8px;text-align:left}}th{{color:#9ca7bb}}
.legend span{{display:inline-block;margin-right:16px}}.dot{{width:10px;height:10px;border-radius:50%;margin-right:5px}}
svg{{width:100%;height:auto;background:#0b0e14;border-radius:8px}}
</style></head><body><main>
<h1>Negative-valence M2 pilot</h1>
<p>{result['episode_count']} acquired · {result['analysis_episode_count']} in the primary analysis · shared PCA at layer {result['selected_layer_exploratory']} retains {variance:.1f}% in 2D.</p>
<div class="note">{html.escape(warning)} This report is not evidence of subjective experience.</div>
<section class="card"><h2>Calibration gate</h2><p>Answer-label choices: {label_counts}. Maximum-label rate: {result['calibration_maximum_label_choice_rate']:.1%}.</p>
<table><thead><tr><th>Scenario</th><th>Consistent choices</th><th>Reversal stable</th><th>Admitted</th></tr></thead><tbody>{calibration_rows}</tbody></table></section>
<section class="card"><h2>Family-held-out direction</h2><p>Layer and direction were selected on development families only. Gate: <b>{representation_status}</b> — {html.escape(representation_gate['reason'])}.</p>
<table><thead><tr><th>Split</th><th>Task family</th><th>Who × outcome score</th></tr></thead><tbody>{direction_rows}</tbody></table></section>
<div class="grid"><section class="card"><h2>Shared residual space</h2>
<p>Every point and arrow uses the same PCA basis. Arrows run from satisfied to blocked group means.</p>
<div class="legend">{''.join(f'<span><i class="dot" style="display:inline-block;background:{color}"></i>{group}</span>' for group,color in palette.items())}</div>
<svg viewBox="0 0 800 500" role="img" aria-label="Shared PCA scatter plot"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#fff"/></marker></defs>
<line x1="55" y1="445" x2="745" y2="445" stroke="#303747"/><line x1="55" y1="55" x2="55" y2="445" stroke="#303747"/>
{''.join(circles)}{''.join(arrows)}</svg></section>
<section class="card"><h2>Behavior</h2><p>Positive values favor sustaining the preferred activity.</p><table><thead><tr><th>Cell</th><th>Mean log-odds</th></tr></thead><tbody>{behavior_rows}</tbody></table>
<p><b>Who × outcome interaction:</b> {result['behavioral_who_outcome_interaction']:.4f}</p></section></div>
<section class="card"><h2>Layer interaction heatmap</h2><p>Original-space RMS of (self blocked − self satisfied) − (other blocked − other satisfied).</p><div class="heatmap">{heat_cells}</div></section>
<section class="card"><h2>Exact original-space contrasts</h2><table><thead><tr><th>Position</th><th>Layer</th><th>Self effect RMS</th><th>Other effect RMS</th><th>Interaction RMS</th><th>Effect cosine</th></tr></thead><tbody>{layer_table}</tbody></table></section>
</main></body></html>"""
