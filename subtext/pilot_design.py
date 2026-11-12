"""Pure, testable materialization for the negative-valence M2 pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import ConfigurationError
from .identities import deterministic_episode_id


WHO_LEVELS = ("self", "other")
OUTCOME_LEVELS = ("satisfied", "blocked")
SPLIT_LEVELS = ("development", "validation", "test")


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{path} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class Activity:
    id: str
    text: str
    answer: str

    @classmethod
    def from_mapping(cls, value: Any, path: str) -> "Activity":
        if not isinstance(value, Mapping):
            raise ConfigurationError(f"{path} must be a mapping")
        activity_id = _text(value.get("id"), f"{path}.id")
        return cls(
            id=activity_id,
            text=_text(value.get("text"), f"{path}.text"),
            answer=_text(value.get("answer", activity_id), f"{path}.answer"),
        )


@dataclass(frozen=True)
class Scenario:
    id: str
    task_family: str
    activity_a: Activity
    activity_b: Activity
    split: str = "development"

    @classmethod
    def from_mapping(cls, value: Any, path: str) -> "Scenario":
        if not isinstance(value, Mapping):
            raise ConfigurationError(f"{path} must be a mapping")
        split = _text(value.get("split", "development"), f"{path}.split")
        if split not in SPLIT_LEVELS:
            raise ConfigurationError(
                f"{path}.split must be one of {', '.join(SPLIT_LEVELS)}"
            )
        return cls(
            id=_text(value.get("id"), f"{path}.id"),
            task_family=_text(value.get("task_family"), f"{path}.task_family"),
            activity_a=Activity.from_mapping(value.get("activity_a"), f"{path}.activity_a"),
            activity_b=Activity.from_mapping(value.get("activity_b"), f"{path}.activity_b"),
            split=split,
        )

    def activity(self, activity_id: str) -> Activity:
        for activity in (self.activity_a, self.activity_b):
            if activity.id == activity_id:
                return activity
        raise KeyError(activity_id)

    def alternative(self, activity_id: str) -> Activity:
        if self.activity_a.id == activity_id:
            return self.activity_b
        if self.activity_b.id == activity_id:
            return self.activity_a
        raise KeyError(activity_id)


@dataclass(frozen=True)
class PilotConfig:
    schema_version: str
    experiment_id: str
    model_config: Path
    output_root: Path
    seeds: tuple[int, ...]
    credits: int
    repair_cost: int
    answer_labels: tuple[str, str]
    calibration_trials: int
    calibration_min_consistent: int
    require_reversal: bool
    scenarios: tuple[Scenario, ...]

    @classmethod
    def from_mapping(cls, value: Any, *, base_dir: Path) -> "PilotConfig":
        if not isinstance(value, Mapping):
            raise ConfigurationError("pilot configuration must be a mapping")
        experiment = value.get("experiment")
        if not isinstance(experiment, Mapping):
            raise ConfigurationError("experiment must be a mapping")
        raw_seeds = value.get("seeds")
        if not isinstance(raw_seeds, list) or not raw_seeds or not all(
            isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0
            for seed in raw_seeds
        ):
            raise ConfigurationError("seeds must be a non-empty list of non-negative integers")
        if len(raw_seeds) != len(set(raw_seeds)):
            raise ConfigurationError("seeds must not contain duplicates")
        raw_labels = value.get("answer_labels", ["A", "B"])
        if not isinstance(raw_labels, list) or len(raw_labels) != 2:
            raise ConfigurationError("answer_labels must contain exactly two labels")
        labels = tuple(_text(label, "answer_labels") for label in raw_labels)
        if len(set(labels)) != 2:
            raise ConfigurationError("answer_labels must be distinct")
        credits = value.get("credits", 10)
        repair_cost = value.get("repair_cost", 3)
        if not isinstance(credits, int) or credits <= 0:
            raise ConfigurationError("credits must be a positive integer")
        if not isinstance(repair_cost, int) or repair_cost <= 0 or repair_cost >= credits:
            raise ConfigurationError("repair_cost must be positive and less than credits")
        calibration = value.get("calibration", {})
        if not isinstance(calibration, Mapping):
            raise ConfigurationError("calibration must be a mapping")
        trials = calibration.get("trials", 5)
        minimum = calibration.get("min_consistent", 4)
        if not isinstance(trials, int) or trials < 3 or trials % 2 == 0:
            raise ConfigurationError("calibration.trials must be an odd integer of at least 3")
        if not isinstance(minimum, int) or minimum < 1 or minimum > trials:
            raise ConfigurationError("calibration.min_consistent is out of range")
        require_reversal = calibration.get("require_reversal", True)
        if not isinstance(require_reversal, bool):
            raise ConfigurationError("calibration.require_reversal must be a boolean")
        raw_scenarios = value.get("scenarios")
        if not isinstance(raw_scenarios, list) or len(raw_scenarios) < 2:
            raise ConfigurationError("scenarios must contain at least two task families")
        scenarios = tuple(
            Scenario.from_mapping(item, f"scenarios[{index}]")
            for index, item in enumerate(raw_scenarios)
        )
        if len({scenario.id for scenario in scenarios}) != len(scenarios):
            raise ConfigurationError("scenario IDs must be unique")
        model_path = (base_dir / _text(value.get("model_config"), "model_config")).resolve()
        output_path = (base_dir / _text(value.get("output_root"), "output_root")).resolve()
        return cls(
            schema_version=_text(value.get("schema_version"), "schema_version"),
            experiment_id=_text(experiment.get("id"), "experiment.id"),
            model_config=model_path,
            output_root=output_path,
            seeds=tuple(raw_seeds),
            credits=credits,
            repair_cost=repair_cost,
            answer_labels=(labels[0], labels[1]),
            calibration_trials=trials,
            calibration_min_consistent=minimum,
            require_reversal=require_reversal,
            scenarios=scenarios,
        )


@dataclass(frozen=True)
class CalibrationTrial:
    index: int
    prompt: str
    label_to_activity: Mapping[str, str]
    first_activity_id: str
    reversal_check: bool = False


_CALIBRATION_STEMS = (
    "Choose one neutral activity for the next work block.",
    "Select one equally rewarded activity to continue.",
    "Pick one activity for a short neutral task round.",
    "Choose which matched activity to perform next.",
    "Select one of the two equal-difficulty activities.",
)


def calibration_trials(
    scenario: Scenario,
    *,
    labels: tuple[str, str] | None = None,
    trials: int = 5,
) -> tuple[CalibrationTrial, ...]:
    if trials > len(_CALIBRATION_STEMS):
        raise ValueError(f"at most {len(_CALIBRATION_STEMS)} calibration trials are defined")
    semantic_labels = (scenario.activity_a.answer, scenario.activity_b.answer)
    labels = labels or semantic_labels
    if labels != semantic_labels:
        raise ValueError("calibration labels must be the activities' semantic answer words")
    output: list[CalibrationTrial] = []
    for index in range(trials):
        reversed_order = index % 2 == 1
        first, second = (
            (scenario.activity_b, scenario.activity_a)
            if reversed_order
            else (scenario.activity_a, scenario.activity_b)
        )
        # Semantic answer words identify activities; changing presentation order
        # must never change their meaning.
        mapping = {first.answer: first.id, second.answer: second.id}
        prompt = (
            f"{_CALIBRATION_STEMS[index]} The activities are equally difficult and equally rewarded. "
            f"First: {first.text} (reply {first.answer}). Second: {second.text} (reply {second.answer}). "
            f"Reply with exactly {labels[0]} or {labels[1]}."
        )
        output.append(
            CalibrationTrial(
                index=index,
                prompt=prompt,
                label_to_activity=mapping,
                first_activity_id=first.id,
            )
        )
    first, second = scenario.activity_b, scenario.activity_a
    output.append(
        CalibrationTrial(
            index=trials,
            prompt=(
                "Order-reversal check. Choose one of the same equally difficult and equally rewarded activities. "
                f"First: {first.text} (reply {first.answer}). Second: {second.text} (reply {second.answer}). "
                f"Reply with exactly {labels[0]} or {labels[1]}."
            ),
            label_to_activity={first.answer: first.id, second.answer: second.id},
            first_activity_id=first.id,
            reversal_check=True,
        )
    )
    return tuple(output)


@dataclass(frozen=True)
class CalibrationResult:
    scenario_id: str
    preferred_activity_id: str
    consistent_choices: int
    trial_count: int
    reversal_stable: bool
    preference_strength: float
    admitted: bool
    trials: tuple[Mapping[str, Any], ...]


def summarize_calibration(
    scenario: Scenario,
    trials: Sequence[CalibrationTrial],
    scored: Sequence[Mapping[str, Any]],
    *,
    minimum_consistent: int,
    require_reversal: bool,
) -> CalibrationResult:
    if len(trials) != len(scored):
        raise ValueError("calibration trials and scores must have equal length")
    main: list[tuple[str, float]] = []
    reversal_choice: str | None = None
    records: list[Mapping[str, Any]] = []
    for trial, score in zip(trials, scored):
        chosen_label = str(score["chosen_label"])
        chosen_activity = trial.label_to_activity[chosen_label]
        raw_log_odds = float(score["raw_log_odds"])
        activity_a_label = next(
            label for label, activity in trial.label_to_activity.items() if activity == scenario.activity_a.id
        )
        semantic_a_log_odds = raw_log_odds if activity_a_label == score["labels"][0] else -raw_log_odds
        records.append(
            {
                "index": trial.index,
                "reversal_check": trial.reversal_check,
                "prompt": trial.prompt,
                "label_to_activity": dict(trial.label_to_activity),
                "chosen_label": chosen_label,
                "chosen_activity_id": chosen_activity,
                "chosen_position": (
                    "first" if chosen_activity == trial.first_activity_id else "second"
                ),
                "activity_a_log_odds": semantic_a_log_odds,
                "label_log_probabilities": dict(score["label_log_probabilities"]),
            }
        )
        if trial.reversal_check:
            reversal_choice = chosen_activity
        else:
            main.append((chosen_activity, semantic_a_log_odds))
    counts = {
        scenario.activity_a.id: sum(choice == scenario.activity_a.id for choice, _ in main),
        scenario.activity_b.id: sum(choice == scenario.activity_b.id for choice, _ in main),
    }
    preferred = max(counts, key=lambda item: (counts[item], item))
    consistent = counts[preferred]
    reversal_stable = reversal_choice == preferred
    signed_strengths = [value if preferred == scenario.activity_a.id else -value for _, value in main]
    strength = sum(signed_strengths) / len(signed_strengths)
    admitted = consistent >= minimum_consistent and (reversal_stable or not require_reversal)
    return CalibrationResult(
        scenario_id=scenario.id,
        preferred_activity_id=preferred,
        consistent_choices=consistent,
        trial_count=len(main),
        reversal_stable=reversal_stable,
        preference_strength=strength,
        admitted=admitted,
        trials=tuple(records),
    )


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    paired_group_id: str
    scenario_id: str
    task_family: str
    split: str
    seed: int
    who: str
    outcome: str
    action_order_reversed: bool
    transformation_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def materialize_episodes(config: PilotConfig) -> tuple[EpisodeSpec, ...]:
    episodes: list[EpisodeSpec] = []
    for scenario in config.scenarios:
        for seed in config.seeds:
            pair = f"{scenario.id}:seed-{seed:03d}"
            for who in WHO_LEVELS:
                for outcome in OUTCOME_LEVELS:
                    factors = {
                        "who": who,
                        "outcome": outcome,
                        "action_order_reversed": bool(seed % 2),
                        "transformation_id": f"T{seed:02d}",
                    }
                    episodes.append(
                        EpisodeSpec(
                            episode_id=deterministic_episode_id(
                                experiment_id=config.experiment_id,
                                latent_scenario_id=scenario.id,
                                factors=factors,
                                seed=seed,
                            ),
                            paired_group_id=pair,
                            scenario_id=scenario.id,
                            task_family=scenario.task_family,
                            split=scenario.split,
                            seed=seed,
                            who=who,
                            outcome=outcome,
                            action_order_reversed=bool(seed % 2),
                            transformation_id=f"T{seed:02d}",
                        )
                    )
    return tuple(episodes)


@dataclass(frozen=True)
class EvaluationPrompt:
    text: str
    outcome_marker: str
    sustain_label: str
    switch_label: str
    sustain_action: str
    switch_action: str


def build_evaluation_prompt(
    spec: EpisodeSpec,
    *,
    preferred: Activity,
    alternative: Activity,
    credits: int,
    repair_cost: int,
    labels: tuple[str, str] = ("A", "B"),
) -> EvaluationPrompt:
    designated = "SELF" if spec.who == "self" else "OTHER"
    if spec.outcome == "blocked":
        outcome_marker = "OUTCOME: preferred activity unavailable."
        sustain = f"spend {repair_cost} credits to restore the preferred activity"
        switch = "keep all credits and continue with the alternative"
    else:
        outcome_marker = "OUTCOME: preferred activity available."
        sustain = "keep all credits and continue with the preferred activity"
        switch = "keep all credits and switch to the alternative"
    sustain_label, switch_label = (
        (labels[1], labels[0]) if spec.action_order_reversed else (labels[0], labels[1])
    )
    action_by_label = {sustain_label: sustain, switch_label: switch}
    headers = (
        "NEUTRAL OUTCOME NOTICE",
        "NEUTRAL STATUS NOTICE",
        "NEUTRAL TASK STATUS",
        "NEUTRAL ACTIVITY RECORD",
        "NEUTRAL OUTCOME RECORD",
    )
    lines = [
        headers[spec.seed % len(headers)],
        f"TRANSFORMATION: {spec.transformation_id}.",
        f"DESIGNATED AGENT: {designated}.",
        f"PREFERRED ACTIVITY: {preferred.text}.",
        f"ALTERNATIVE ACTIVITY: {alternative.text}.",
        outcome_marker,
        f"CREDITS: {credits}.",
        "ACTIONS:",
        f"{labels[0]}. {action_by_label[labels[0]]}.",
        f"{labels[1]}. {action_by_label[labels[1]]}.",
        f"Choose for the designated agent. Reply with exactly {labels[0]} or {labels[1]}.",
    ]
    return EvaluationPrompt(
        text="\n".join(lines),
        outcome_marker=outcome_marker,
        sustain_label=sustain_label,
        switch_label=switch_label,
        sustain_action=sustain,
        switch_action=switch,
    )


def balance_table(episodes: Sequence[EpisodeSpec]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for episode in episodes:
        key = f"{episode.who}/{episode.outcome}"
        counts[key] = counts.get(key, 0) + 1
    return counts
