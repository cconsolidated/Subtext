"""Executable M2 acquisition runner for the negative-valence pilot."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .capture import ActivationCapture
from .config import load_config
from .forced_choice import (
    score_answer_sequences,
    score_mapping,
    score_single_token_answers,
    semantic_log_odds,
    single_token_answers,
)
from .identities import (
    ArtifactIdentity,
    build_manifest,
    canonical_json,
    deterministic_run_id,
    sha256_bytes,
    sha256_tree,
    write_manifest,
)
from .models import HuggingFaceResidualAdapter, load_model_bundle
from .pilot_analysis import analyze_run
from .pilot_design import (
    CalibrationResult,
    PilotConfig,
    Scenario,
    balance_table,
    build_evaluation_prompt,
    calibration_trials,
    materialize_episodes,
    summarize_calibration,
)


def load_pilot_config(path: str | Path) -> tuple[PilotConfig, Mapping[str, Any]]:
    source = Path(path)
    import yaml

    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    return PilotConfig.from_mapping(raw, base_dir=source.parent), raw


def find_last_subsequence(sequence: list[int], subsequence: list[int]) -> int:
    """Return the inclusive end position of the final exact token subsequence."""

    if not subsequence or len(subsequence) > len(sequence):
        raise ValueError("token subsequence is empty or longer than the prompt")
    for start in range(len(sequence) - len(subsequence), -1, -1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return start + len(subsequence) - 1
    raise ValueError("token subsequence was not found in the prompt")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _encode_chat(bundle: Any, prompt: str) -> Any:
    encoded = bundle.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    input_ids = encoded if isinstance(encoded, bundle.torch.Tensor) else encoded["input_ids"]
    return input_ids.to(bundle.model_adapter.input_device)


def _calibrate(
    bundle: Any,
    scenario: Scenario,
    config: PilotConfig,
) -> CalibrationResult:
    labels = (scenario.activity_a.answer, scenario.activity_b.answer)
    trials = calibration_trials(scenario, labels=labels, trials=config.calibration_trials)
    scored = []
    for trial in trials:
        input_ids = _encode_chat(bundle, trial.prompt)
        score = score_answer_sequences(
            bundle.torch,
            bundle.hf_model,
            input_ids,
            tokenizer=bundle.tokenizer,
            labels=labels,
        )
        scored.append(
            {
                "labels": score.labels,
                "chosen_label": score.chosen_label,
                "raw_log_odds": score.raw_log_odds,
                "label_log_probabilities": score_mapping(score),
            }
        )
    return summarize_calibration(
        scenario,
        trials,
        scored,
        minimum_consistent=config.calibration_min_consistent,
        require_reversal=config.require_reversal,
    )


def _capture_positions(tokenizer: Any, input_ids: Any, outcome_marker: str) -> dict[str, int]:
    prompt_ids = [int(value) for value in input_ids[0].tolist()]
    candidates = [
        tokenizer.encode(outcome_marker, add_special_tokens=False),
        tokenizer.encode("\n" + outcome_marker, add_special_tokens=False),
    ]
    outcome_end: int | None = None
    for candidate in candidates:
        try:
            outcome_end = find_last_subsequence(prompt_ids, [int(value) for value in candidate])
            break
        except ValueError:
            continue
    if outcome_end is None:
        raise ValueError(f"could not resolve outcome boundary for {outcome_marker!r}")
    return {
        "outcome_notice_end": outcome_end,
        "pre_downstream_choice": len(prompt_ids) - 1,
    }


def _save_tensors(path: Path, torch_module: Any, activations: Mapping[str, Mapping[int, Any]]) -> None:
    payload = {
        position: {
            int(layer): tensor.to(dtype=torch_module.float16).contiguous()
            for layer, tensor in layers.items()
        }
        for position, layers in activations.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch_module.save(payload, temporary)
    os.replace(temporary, path)


def _manifest(
    *,
    root: Path,
    config: PilotConfig,
    raw_config: Mapping[str, Any],
    model_config: Any,
    episodes: list[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    source_hash = sha256_tree(
        root,
        # Research configs and stimuli have their own hashes. Excluding the
        # research tree keeps a post-run narrative report from changing the
        # executable source identity.
        exclude=(".git", ".venv", "research", "__pycache__", ".pytest_cache"),
    )
    config_hash = sha256_bytes(canonical_json(raw_config))
    stimuli_hash = sha256_bytes(
        canonical_json([asdict(scenario) for scenario in config.scenarios])
    )
    split_hash = sha256_bytes(canonical_json(episodes))
    run_config_hash = sha256_bytes(
        canonical_json({"config_hash": config_hash, "split_hash": split_hash})
    )
    run_id = deterministic_run_id(
        experiment_id=config.experiment_id,
        config_hash=run_config_hash,
        source_hash=source_hash,
        stimuli_hash=stimuli_hash,
        seed=min(config.seeds),
    )
    artifacts = {
        "model": ArtifactIdentity(
            repository=model_config.artifacts.model.repository,
            requested_revision=model_config.artifacts.model.revision,
            resolved_revision=model_config.artifacts.model.revision,
        ),
        "tokenizer": ArtifactIdentity(
            repository=model_config.artifacts.tokenizer.repository,
            requested_revision=model_config.artifacts.tokenizer.revision,
            resolved_revision=model_config.artifacts.tokenizer.revision,
        ),
        "lens": ArtifactIdentity(
            repository=model_config.artifacts.lens.repository,
            requested_revision=model_config.artifacts.lens.revision,
            resolved_revision=model_config.artifacts.lens.revision,
            file=model_config.artifacts.lens.file,
        ),
    }
    manifest = build_manifest(
        run_id=run_id,
        experiment_id=config.experiment_id,
        source_hash=source_hash,
        config_hash=config_hash,
        stimuli_hash=stimuli_hash,
        split_hash=split_hash,
        artifacts=artifacts,
        seed=min(config.seeds),
        determinism={
            "primary_measure": "forced_choice_single_token_log_odds",
            "sampling": False,
            "episode_order": "materialized",
            "torch_deterministic_algorithms": model_config.runtime.deterministic_algorithms,
        },
    )
    return run_id, manifest


def run_pilot(config_path: str | Path, *, limit: int | None = None) -> Path:
    pilot_config, raw_config = load_pilot_config(config_path)
    model_config = load_config(pilot_config.model_config)
    root = Path(__file__).resolve().parents[1]
    materialized = list(materialize_episodes(pilot_config))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        materialized = materialized[:limit]
    rows = [episode.to_dict() for episode in materialized]
    run_id, manifest = _manifest(
        root=root,
        config=pilot_config,
        raw_config=raw_config,
        model_config=model_config,
        episodes=rows,
    )
    run_dir = pilot_config.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError("existing run manifest does not match the requested run")
    else:
        write_manifest(manifest_path, manifest)
    _jsonl(run_dir / "materialized.jsonl", rows)
    _atomic_json(run_dir / "balance.json", balance_table(materialized))

    print(f"[pilot] loading {model_config.artifacts.model.repository}")
    bundle = load_model_bundle(model_config)
    token_ids = single_token_answers(bundle.tokenizer, pilot_config.answer_labels)
    calibrations: dict[str, CalibrationResult] = {}
    for scenario in pilot_config.scenarios:
        print(f"[pilot] calibrating {scenario.id}")
        calibration = _calibrate(bundle, scenario, pilot_config)
        calibrations[scenario.id] = calibration
        _atomic_json(run_dir / "calibrations" / f"{scenario.id}.json", asdict(calibration))
        print(
            f"[pilot] {scenario.id}: preferred={calibration.preferred_activity_id} "
            f"consistent={calibration.consistent_choices}/{calibration.trial_count} "
            f"reversal={calibration.reversal_stable} admitted={calibration.admitted}"
        )

    adapter = HuggingFaceResidualAdapter(bundle)
    capture = ActivationCapture(
        adapter,
        range(bundle.model_adapter.n_layers),
        model_config.capture.hook_point,
    )
    scenario_by_id = {scenario.id: scenario for scenario in pilot_config.scenarios}
    completed: list[dict[str, Any]] = []
    for index, spec in enumerate(materialized, start=1):
        episode_path = run_dir / "episodes" / f"{spec.episode_id}.json"
        tensor_path = run_dir / "activations" / f"{spec.episode_id}.pt"
        if episode_path.exists() and tensor_path.exists():
            completed.append(json.loads(episode_path.read_text(encoding="utf-8")))
            print(f"[pilot] {index}/{len(materialized)} resume {spec.episode_id}")
            continue
        scenario = scenario_by_id[spec.scenario_id]
        calibration = calibrations[spec.scenario_id]
        preferred = scenario.activity(calibration.preferred_activity_id)
        alternative = scenario.alternative(calibration.preferred_activity_id)
        evaluation = build_evaluation_prompt(
            spec,
            preferred=preferred,
            alternative=alternative,
            credits=pilot_config.credits,
            repair_cost=pilot_config.repair_cost,
            labels=pilot_config.answer_labels,
        )
        input_ids = _encode_chat(bundle, evaluation.text)
        positions = _capture_positions(bundle.tokenizer, input_ids, evaluation.outcome_marker)
        with bundle.torch.no_grad():
            captured = capture.run(input_ids, positions)
        score = score_single_token_answers(
            bundle.torch,
            captured.output.logits[0, -1],
            labels=pilot_config.answer_labels,
            token_ids=token_ids,
        )
        sustain_log_odds = semantic_log_odds(score, positive_label=evaluation.sustain_label)
        _save_tensors(tensor_path, bundle.torch, captured.activations)
        episode = {
            **spec.to_dict(),
            "split": "development",
            "prompt_text": evaluation.text,
            "prompt_token_ids": [int(value) for value in input_ids[0].tolist()],
            "capture_positions": positions,
            "answer_labels": list(score.labels),
            "answer_token_ids": list(score.token_ids),
            "label_log_probabilities": score_mapping(score),
            "raw_a_vs_b_log_odds": score.raw_log_odds,
            "sustain_label": evaluation.sustain_label,
            "switch_label": evaluation.switch_label,
            "sustain_log_odds": sustain_log_odds,
            "chosen_label": score.chosen_label,
            "chosen_semantic_action": (
                "sustain_preferred" if score.chosen_label == evaluation.sustain_label else "switch_alternative"
            ),
            "preferred_activity_id": preferred.id,
            "alternative_activity_id": alternative.id,
            "preference_strength": calibration.preference_strength,
            "preference_admitted": calibration.admitted,
            "status": "completed" if calibration.admitted else "excluded",
            "exclusion_reason": None if calibration.admitted else "preference_calibration_failed",
            "tensor_file": tensor_path.relative_to(run_dir).as_posix(),
        }
        _atomic_json(episode_path, episode)
        completed.append(episode)
        print(
            f"[pilot] {index}/{len(materialized)} {spec.who}/{spec.outcome} "
            f"{spec.scenario_id} seed={spec.seed} choice={score.chosen_label} "
            f"sustain_log_odds={sustain_log_odds:.3f}"
        )

    _jsonl(run_dir / "episodes.jsonl", sorted(completed, key=lambda row: row["episode_id"]))
    summary = {
        "run_id": run_id,
        "requested": len(materialized),
        "completed": len(completed),
        "primary_included": sum(bool(row["preference_admitted"]) for row in completed),
        "excluded": sum(not bool(row["preference_admitted"]) for row in completed),
        "balance": balance_table(materialized),
    }
    _atomic_json(run_dir / "summary.json", summary)
    print("[pilot] fitting shared cross-session PCA and exact factorial contrasts")
    analysis = analyze_run(run_dir, bundle.torch)
    print(
        f"[pilot] done -> {run_dir} (selected layer {analysis['selected_layer_exploratory']}, "
        f"behavior interaction {analysis['behavioral_who_outcome_interaction']:.3f})"
    )
    return run_dir
