"""Calibration-only task-family screen with explicit position-bias gates."""

from __future__ import annotations

from dataclasses import asdict
import html
import json
from pathlib import Path
from typing import Any, Mapping

from .config import load_config
from .identities import canonical_json, sha256_bytes, sha256_tree, stable_id
from .models import load_model_bundle
from .pilot_runner import calibrate_scenario, load_pilot_config


def screen_gate(
    results: list[Mapping[str, Any]],
    *,
    min_admitted_rate: float,
    min_admitted_families: int,
    max_position_choice_rate: float,
    require_split_coverage: bool,
) -> dict[str, Any]:
    """Evaluate Gate A without looking at factorial outcomes."""

    if not results:
        raise ValueError("calibration screen requires at least one result")
    admitted = [result for result in results if bool(result["admitted"])]
    positions = [
        trial["chosen_position"]
        for result in results
        for trial in result["trials"]
    ]
    first_rate = positions.count("first") / len(positions)
    maximum_position_rate = max(first_rate, 1.0 - first_rate)
    admitted_splits = sorted({str(result["split"]) for result in admitted})
    split_coverage = all(
        split in admitted_splits for split in ("development", "validation", "test")
    )
    admitted_rate = len(admitted) / len(results)
    checks = {
        "admitted_rate": admitted_rate >= min_admitted_rate,
        "admitted_family_count": len(admitted) >= min_admitted_families,
        "position_balance": maximum_position_rate <= max_position_choice_rate,
        "split_coverage": split_coverage or not require_split_coverage,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "admitted_count": len(admitted),
        "scenario_count": len(results),
        "admitted_rate": admitted_rate,
        "first_position_choice_rate": first_rate,
        "maximum_position_choice_rate": maximum_position_rate,
        "admitted_splits": admitted_splits,
    }


def _render_html(payload: Mapping[str, Any]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['scenario_id'])}</td>"
        f"<td>{html.escape(item['split'])}</td>"
        f"<td>{item['consistent_choices']}/{item['trial_count']}</td>"
        f"<td>{'yes' if item['reversal_stable'] else 'no'}</td>"
        f"<td>{item['preference_strength']:.3f}</td>"
        f"<td>{'yes' if item['admitted'] else 'no'}</td>"
        "</tr>"
        for item in payload["results"]
    )
    gate = payload["gate"]
    badge = "PASS" if gate["passed"] else "HOLD"
    color = "#52d273" if gate["passed"] else "#ffb454"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Calibration screen</title>
<style>body{{background:#090b10;color:#e8ecf3;font:15px system-ui;margin:0}}main{{max-width:1050px;margin:auto;padding:36px}}
.card{{background:#111621;border:1px solid #273043;border-radius:12px;padding:20px;margin:18px 0}}.badge{{color:{color};font-weight:800}}
table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #293143}}th{{color:#9ba8be}}</style></head>
<body><main><h1>Calibration-only family screen</h1><div class="card"><span class="badge">{badge}</span>
<p>{gate['admitted_count']}/{gate['scenario_count']} families admitted ({gate['admitted_rate']:.1%}).
Maximum presentation-position choice rate: {gate['maximum_position_choice_rate']:.1%}.</p>
<p>No blocked/satisfied factorial outcomes were inspected during this screen.</p></div>
<div class="card"><table><thead><tr><th>Family</th><th>Frozen split</th><th>Consistent</th><th>Reversal</th><th>Strength</th><th>Admitted</th></tr></thead>
<tbody>{rows}</tbody></table></div><p>This is an operational preference screen, not evidence of experience or welfare.</p></main></body></html>"""


def run_calibration_screen(config_path: str | Path) -> Path:
    """Run only neutral preference calibration and write an auditable report."""

    config, raw = load_pilot_config(config_path)
    model_config = load_config(config.model_config)
    root = Path(__file__).resolve().parents[1]
    source_hash = sha256_tree(
        root,
        exclude=(".git", ".venv", "research", "__pycache__", ".pytest_cache"),
    )
    config_hash = sha256_bytes(canonical_json(raw))
    screen_id = stable_id(
        "screen",
        {
            "experiment_id": config.experiment_id,
            "config_hash": config_hash,
            "source_hash": source_hash,
            "model_revision": model_config.artifacts.model.revision,
        },
    )
    output = config.output_root / screen_id
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "0.1",
        "screen_id": screen_id,
        "experiment_id": config.experiment_id,
        "config_hash": config_hash,
        "source_hash": source_hash,
        "model": asdict(model_config.artifacts.model),
        "factorial_outcomes_accessed": False,
    }
    manifest_path = output / "screen_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError("existing calibration-screen manifest does not match")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"[screen] loading {model_config.artifacts.model.repository}")
    bundle = load_model_bundle(model_config)
    results: list[dict[str, Any]] = []
    for index, scenario in enumerate(config.scenarios, start=1):
        calibration = calibrate_scenario(bundle, scenario, config)
        row = {**asdict(calibration), "split": scenario.split, "task_family": scenario.task_family}
        results.append(row)
        print(
            f"[screen] {index}/{len(config.scenarios)} {scenario.id}: "
            f"{calibration.consistent_choices}/{calibration.trial_count} "
            f"reversal={calibration.reversal_stable} admitted={calibration.admitted}"
        )

    raw_gate = raw.get("screen_gate", {})
    gate = screen_gate(
        results,
        min_admitted_rate=float(raw_gate.get("min_admitted_rate", 0.60)),
        min_admitted_families=int(raw_gate.get("min_admitted_families", 3)),
        max_position_choice_rate=float(raw_gate.get("max_position_choice_rate", 0.75)),
        require_split_coverage=bool(raw_gate.get("require_split_coverage", True)),
    )
    payload = {
        "schema_version": "0.1",
        "screen_id": screen_id,
        "gate": gate,
        "frozen_split_assignments": {
            item["scenario_id"]: item["split"] for item in results
        },
        "admitted_scenarios": [
            item["scenario_id"] for item in results if item["admitted"]
        ],
        "results": results,
    }
    (output / "screen_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "report.html").write_text(_render_html(payload), encoding="utf-8")
    print(f"[screen] done -> {output} gate={'PASS' if gate['passed'] else 'HOLD'}")
    return output
