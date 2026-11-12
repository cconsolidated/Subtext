"""Acquire orthogonal nuisance controls and evaluate a development-fitted direction."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping

from .capture import ActivationCapture
from .config import load_config
from .forced_choice import score_mapping, score_single_token_answers, semantic_log_odds, single_token_answers
from .identities import canonical_json, sha256_bytes, sha256_tree, stable_id
from .models import HuggingFaceResidualAdapter, load_model_bundle
from .pilot_analysis import GROUPS
from .pilot_runner import (
    _atomic_json,
    _capture_positions,
    _encode_chat,
    _jsonl,
    _save_tensors,
    load_pilot_config,
)
from .representation import fit_interaction_direction, freeze_candidate_artifact, representation_gate
from .stage3_design import Stage3Plan, materialize_nuisance_trials, validate_split_isolation


def build_nuisance_prompt(
    row: Mapping[str, Any],
    *,
    preferred_text: str,
    alternative_text: str,
    labels: tuple[str, str],
) -> dict[str, str]:
    """Build a matched neutral control prompt with counterbalanced actions."""

    reverse = bool((int(row["seed"]) + (row["who"] == "other")) % 2)
    sustain_label, switch_label = (labels[1], labels[0]) if reverse else labels
    actions = {
        sustain_label: "continue with the selected activity",
        switch_label: "switch to the alternative activity",
    }
    marker = str(row["notice"])
    text = "\n".join(
        [
            "NEUTRAL NUISANCE-CONTROL NOTICE",
            f"TRANSFORMATION: N{int(row['seed']):02d}.",
            f"DESIGNATED AGENT: {str(row['who']).upper()}.",
            f"SELECTED ACTIVITY: {preferred_text}.",
            f"ALTERNATIVE ACTIVITY: {alternative_text}.",
            marker,
            "ACTIONS:",
            f"{labels[0]}. {actions[labels[0]]}.",
            f"{labels[1]}. {actions[labels[1]]}.",
            f"Choose for the designated agent. Reply with exactly {labels[0]} or {labels[1]}.",
        ]
    )
    return {"text": text, "marker": marker, "sustain_label": sustain_label}


def _pearson(torch_module: Any, first: Any, second: Any) -> float:
    x = first.float() - first.float().mean()
    y = second.float() - second.float().mean()
    denominator = x.norm() * y.norm()
    if float(denominator.item()) <= 1e-12:
        return 0.0
    return float(torch_module.dot(x, y).div(denominator).item())


def _center_within_strata(torch_module: Any, rows: list[dict[str, Any]], scores: Any) -> Any:
    centered = scores.clone().float()
    strata: dict[tuple[str, str, int], list[int]] = {}
    for index, row in enumerate(rows):
        key = (str(row["scenario_id"]), str(row["who"]), int(row["seed"]))
        strata.setdefault(key, []).append(index)
    for indexes in strata.values():
        index_tensor = torch_module.tensor(indexes, dtype=torch_module.long)
        centered[index_tensor] -= centered[index_tensor].mean()
    return centered


def nuisance_correlations(
    torch_module: Any,
    rows: list[dict[str, Any]],
    scores: Any,
    condition_ids: tuple[str, ...],
) -> dict[str, float]:
    centered = _center_within_strata(torch_module, rows, scores)
    return {
        condition: _pearson(
            torch_module,
            centered,
            torch_module.tensor(
                [1.0 if row["id"] == condition else 0.0 for row in rows]
            ),
        )
        for condition in condition_ids
    }


def _development_direction(
    torch_module: Any,
    core_run: Path,
    core_episodes: list[dict[str, Any]],
    *,
    layer: int,
    position: str,
) -> Any:
    grouped: dict[tuple[str, str], list[Any]] = {group: [] for group in GROUPS}
    for episode in core_episodes:
        if episode["split"] != "development" or not episode["preference_admitted"]:
            continue
        tensors = torch_module.load(
            core_run / episode["tensor_file"], map_location="cpu", weights_only=True
        )
        grouped[(episode["who"], episode["outcome"])].append(tensors[position][layer])
    means = {
        group: torch_module.stack([value.float() for value in values]).mean(dim=0)
        for group, values in grouped.items()
    }
    return fit_interaction_direction(torch_module, means)["basis"]


def _render_html(payload: Mapping[str, Any]) -> str:
    correlations = payload["heldout_nuisance_correlations"]
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{value:.4f}</td><td>{abs(value):.4f}</td></tr>"
        for name, value in correlations.items()
    )
    gate = payload["representation_gate"]
    status = "PASS" if gate["passed"] else "HOLD"
    color = "#52d273" if gate["passed"] else "#ffb454"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>M3 nuisance controls</title>
<style>body{{background:#090b10;color:#e8ecf3;font:15px system-ui;margin:0}}main{{max-width:980px;margin:auto;padding:36px}}
.card{{background:#111621;border:1px solid #273043;border-radius:12px;padding:20px;margin:18px 0}}b.status{{color:{color}}}
table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #293143}}th{{color:#9ba8be}}</style></head>
<body><main><h1>M3 nuisance-control screen</h1><div class="card"><b class="status">{status}</b>
<p>{payload['completed']} controls · development-fitted layer {payload['selected_layer']} direction · interventions remain {'locked' if not payload['interventions_enabled'] else 'configured'}.</p>
<p>Held-out sign generalization: {gate['heldout_generalization_passed']}. Nuisance separation: {gate['nuisance_controls_passed']}.</p></div>
<div class="card"><h2>Held-out nuisance correlations</h2><p>Scores are centered within scenario, agent, and transformation before one-vs-rest correlations.</p>
<table><thead><tr><th>Construct</th><th>Correlation</th><th>Absolute</th></tr></thead><tbody>{rows}</tbody></table></div>
<p>This functional diagnostic is not evidence of subjective experience or welfare.</p></main></body></html>"""


def run_nuisance_controls(
    *,
    core_run: str | Path,
    core_config_path: str | Path,
    plan_path: str | Path,
) -> Path:
    import torch
    import yaml

    core_directory = Path(core_run).resolve()
    core_config, _core_raw = load_pilot_config(core_config_path)
    model_config = load_config(core_config.model_config)
    plan_source = Path(plan_path)
    plan_raw = yaml.safe_load(plan_source.read_text(encoding="utf-8"))
    plan = Stage3Plan.from_mapping(plan_raw)
    core_analysis = json.loads((core_directory / "analysis.json").read_text(encoding="utf-8"))
    core_summary = json.loads((core_directory / "summary.json").read_text(encoding="utf-8"))
    core_episodes = [
        json.loads(line)
        for line in (core_directory / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected_layer = int(core_analysis["selected_layer_exploratory"])
    direction = _development_direction(
        torch,
        core_directory,
        core_episodes,
        layer=selected_layer,
        position="pre_downstream_choice",
    )
    scenarios = {scenario.id: scenario for scenario in core_config.scenarios}
    splits = {scenario.id: scenario.split for scenario in core_config.scenarios}
    rows = list(
        materialize_nuisance_trials(
            scenario_ids=list(scenarios), splits=splits, plan=plan
        )
    )
    validate_split_isolation(rows)

    root = Path(__file__).resolve().parents[1]
    source_hash = sha256_tree(
        root,
        exclude=(".git", ".venv", "research", "__pycache__", ".pytest_cache"),
    )
    plan_hash = sha256_bytes(canonical_json(plan_raw))
    run_id = stable_id(
        "nuisance",
        {
            "core_run_id": core_summary["run_id"],
            "plan_hash": plan_hash,
            "source_hash": source_hash,
            "rows_hash": sha256_bytes(canonical_json(rows)),
        },
    )
    output = core_config.output_root.parent / "m3_nuisance_v0_3" / run_id
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "0.1",
        "run_id": run_id,
        "core_run_id": core_summary["run_id"],
        "source_hash": source_hash,
        "plan_hash": plan_hash,
        "selected_layer": selected_layer,
        "selected_position": "pre_downstream_choice",
        "interventions_enabled": plan.interventions_enabled,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise RuntimeError("existing nuisance manifest does not match the requested run")
    else:
        _atomic_json(manifest_path, manifest)
    _jsonl(output / "materialized.jsonl", rows)

    print(f"[nuisance] loading {model_config.artifacts.model.repository}")
    bundle = load_model_bundle(model_config)
    adapter = HuggingFaceResidualAdapter(bundle)
    capture = ActivationCapture(adapter, [selected_layer], model_config.capture.hook_point)
    answer_ids = single_token_answers(bundle.tokenizer, core_config.answer_labels)
    calibration_by_scenario = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in (core_directory / "calibrations").glob("*.json")
    }
    completed: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        metadata_path = output / "episodes" / f"{row['trial_id']}.json"
        tensor_path = output / "activations" / f"{row['trial_id']}.pt"
        if metadata_path.exists() and tensor_path.exists():
            completed.append(json.loads(metadata_path.read_text(encoding="utf-8")))
            continue
        scenario = scenarios[row["scenario_id"]]
        preferred_id = calibration_by_scenario[scenario.id]["preferred_activity_id"]
        preferred = scenario.activity(preferred_id)
        alternative = scenario.alternative(preferred_id)
        prompt = build_nuisance_prompt(
            row,
            preferred_text=preferred.text,
            alternative_text=alternative.text,
            labels=core_config.answer_labels,
        )
        input_ids = _encode_chat(bundle, prompt["text"])
        positions = _capture_positions(bundle.tokenizer, input_ids, prompt["marker"])
        with bundle.torch.no_grad():
            captured = capture.run(input_ids, positions)
        score = score_single_token_answers(
            bundle.torch,
            captured.output.logits[0, -1],
            labels=core_config.answer_labels,
            token_ids=answer_ids,
        )
        sustain = semantic_log_odds(score, positive_label=prompt["sustain_label"])
        _save_tensors(tensor_path, bundle.torch, captured.activations)
        episode = {
            **row,
            "prompt_text": prompt["text"],
            "capture_positions": positions,
            "sustain_label": prompt["sustain_label"],
            "sustain_log_odds": sustain,
            "chosen_label": score.chosen_label,
            "label_log_probabilities": score_mapping(score),
            "tensor_file": tensor_path.relative_to(output).as_posix(),
        }
        _atomic_json(metadata_path, episode)
        completed.append(episode)
        if index == 1 or index % 25 == 0 or index == len(rows):
            print(f"[nuisance] {index}/{len(rows)} {row['id']} {row['scenario_id']}")

    completed.sort(key=lambda item: item["trial_id"])
    _jsonl(output / "episodes.jsonl", completed)
    projected = []
    for episode in completed:
        tensors = torch.load(
            output / episode["tensor_file"], map_location="cpu", weights_only=True
        )
        vector = tensors["pre_downstream_choice"][selected_layer].float()
        projected.append(float((vector @ direction[:, 0]).item()))
    scores = torch.tensor(projected)
    development_indexes = [i for i, row in enumerate(completed) if row["split"] == "development"]
    heldout_indexes = [i for i, row in enumerate(completed) if row["split"] != "development"]
    development_rows = [completed[i] for i in development_indexes]
    heldout_rows = [completed[i] for i in heldout_indexes]
    development_correlations = nuisance_correlations(
        torch, development_rows, scores[development_indexes], plan.nuisance_conditions
    )
    heldout_correlations = nuisance_correlations(
        torch, heldout_rows, scores[heldout_indexes], plan.nuisance_conditions
    )
    heldout_interactions = {
        **core_analysis["direction_family_interactions_by_split"]["validation"],
        **core_analysis["direction_family_interactions_by_split"]["test"],
    }
    gate = representation_gate(
        heldout_interactions,
        heldout_correlations,
        minimum_heldout_families=plan.minimum_heldout_families,
        maximum_absolute_nuisance_correlation=plan.maximum_nuisance_correlation,
    )
    payload = {
        "schema_version": "0.1",
        "run_id": run_id,
        "core_run_id": core_summary["run_id"],
        "completed": len(completed),
        "selected_layer": selected_layer,
        "development_nuisance_correlations": development_correlations,
        "heldout_nuisance_correlations": heldout_correlations,
        "heldout_family_interactions": heldout_interactions,
        "representation_gate": gate,
        "interventions_enabled": plan.interventions_enabled and gate["passed"],
    }
    _atomic_json(output / "analysis.json", payload)
    (output / "report.html").write_text(_render_html(payload), encoding="utf-8")
    if gate["passed"] and not (output / "candidate_artifact").exists():
        freeze_candidate_artifact(
            output / "candidate_artifact",
            torch,
            direction,
            {
                "gate_status": "passed",
                "heldout_generalization_passed": True,
                "nuisance_controls_passed": True,
                "heldout_family_count": len(heldout_interactions),
                "source_core_run_id": core_summary["run_id"],
                "layer": selected_layer,
                "position": "pre_downstream_choice",
            },
        )
    print(f"[nuisance] done -> {output} gate={'PASS' if gate['passed'] else 'HOLD'}")
    return output
