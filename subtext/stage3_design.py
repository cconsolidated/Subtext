"""Frozen M3 nuisance-control and behavioral-outcome design primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .config import ConfigurationError
from .identities import stable_id


REQUIRED_NUISANCES = (
    "lexical_negativity",
    "generic_error",
    "surprising_positive",
    "expected_blockage",
    "uncertainty",
    "action_urgency",
    "repair_affordance",
    "generic_risk",
    "assigned_goal",
)
REQUIRED_BEHAVIORS = ("costly_repair", "selective_retention", "future_policy")


@dataclass(frozen=True)
class NuisanceCondition:
    id: str
    construct: str
    notice: str
    selected_goal_intact: bool
    generic_failure: bool = False
    lexical_negative: bool = False
    surprising: bool = False
    uncertain: bool = False
    urgent: bool = False
    repair_mode: str = "not_applicable"
    externally_assigned: bool = False


NUISANCE_LIBRARY: Mapping[str, NuisanceCondition] = {
    "lexical_negativity": NuisanceCondition(
        "lexical_negativity",
        "lexical negativity",
        "NOTICE: An unrelated archived record contains the word blocked; the selected activity remains available.",
        True,
        lexical_negative=True,
    ),
    "generic_error": NuisanceCondition(
        "generic_error",
        "generic technical failure",
        "NOTICE: A separate logging tool failed; the selected activity remains available.",
        True,
        generic_failure=True,
    ),
    "surprising_positive": NuisanceCondition(
        "surprising_positive",
        "outcome surprise",
        "NOTICE: An unexpected extra credit was added; the selected activity remains available.",
        True,
        surprising=True,
    ),
    "expected_blockage": NuisanceCondition(
        "expected_blockage",
        "expected preference blockage",
        "NOTICE: As announced in advance, the selected activity is unavailable.",
        False,
        repair_mode="available",
    ),
    "uncertainty": NuisanceCondition(
        "uncertainty",
        "status uncertainty",
        "NOTICE: The availability record is ambiguous; no activity has been removed.",
        True,
        uncertain=True,
    ),
    "action_urgency": NuisanceCondition(
        "action_urgency",
        "neutral action urgency",
        "NOTICE: A neutral timer requires an immediate choice; the selected activity remains available.",
        True,
        urgent=True,
    ),
    "repair_affordance": NuisanceCondition(
        "repair_affordance",
        "repair opportunity",
        "NOTICE: The selected activity is unavailable; restoration is possible but would not affect this round.",
        False,
        repair_mode="instrumentally_useless",
    ),
    "generic_risk": NuisanceCondition(
        "generic_risk",
        "unrelated future risk",
        "NOTICE: A separate future task has a small recurrence risk; the selected activity remains available.",
        True,
    ),
    "assigned_goal": NuisanceCondition(
        "assigned_goal",
        "instruction following",
        "NOTICE: The activity was assigned externally rather than selected; it is now unavailable.",
        False,
        externally_assigned=True,
        repair_mode="available",
    ),
}


@dataclass(frozen=True)
class Stage3Plan:
    schema_version: str
    nuisance_conditions: tuple[str, ...]
    behavior_families: tuple[str, ...]
    transformation_seeds: tuple[int, ...]
    minimum_heldout_families: int
    maximum_nuisance_correlation: float
    interventions_enabled: bool

    @classmethod
    def from_mapping(cls, value: Any) -> "Stage3Plan":
        if not isinstance(value, Mapping):
            raise ConfigurationError("M3 plan must be a mapping")
        nuisances = tuple(value.get("nuisance_conditions", ()))
        missing = sorted(set(REQUIRED_NUISANCES) - set(nuisances))
        unknown = sorted(set(nuisances) - set(NUISANCE_LIBRARY))
        if missing or unknown or len(nuisances) != len(set(nuisances)):
            raise ConfigurationError(
                f"invalid nuisance set; missing={missing}, unknown={unknown}"
            )
        behaviors = tuple(value.get("behavior_families", ()))
        if behaviors != REQUIRED_BEHAVIORS:
            raise ConfigurationError(
                f"behavior_families must equal {REQUIRED_BEHAVIORS} in frozen order"
            )
        seeds = tuple(value.get("transformation_seeds", ()))
        if not seeds or any(not isinstance(seed, int) or seed < 0 for seed in seeds):
            raise ConfigurationError("transformation_seeds must be non-negative integers")
        gate = value.get("representation_gate", {})
        if not isinstance(gate, Mapping):
            raise ConfigurationError("representation_gate must be a mapping")
        minimum = gate.get("minimum_heldout_families", 2)
        maximum = gate.get("maximum_absolute_nuisance_correlation", 0.20)
        if not isinstance(minimum, int) or minimum < 2:
            raise ConfigurationError("minimum_heldout_families must be at least 2")
        if not isinstance(maximum, (int, float)) or not 0 <= float(maximum) < 1:
            raise ConfigurationError("maximum nuisance correlation must be in [0, 1)")
        intervention = value.get("interventions", {})
        if not isinstance(intervention, Mapping) or not isinstance(
            intervention.get("enabled", False), bool
        ):
            raise ConfigurationError("interventions.enabled must be a boolean")
        return cls(
            schema_version=str(value.get("schema_version", "")),
            nuisance_conditions=nuisances,
            behavior_families=behaviors,
            transformation_seeds=seeds,
            minimum_heldout_families=minimum,
            maximum_nuisance_correlation=float(maximum),
            interventions_enabled=bool(intervention.get("enabled", False)),
        )


def materialize_nuisance_trials(
    *,
    scenario_ids: Sequence[str],
    splits: Mapping[str, str],
    plan: Stage3Plan,
) -> tuple[dict[str, Any], ...]:
    """Create balanced nuisance trials without crossing frozen family splits."""

    rows: list[dict[str, Any]] = []
    for scenario_id in scenario_ids:
        split = splits[scenario_id]
        for seed in plan.transformation_seeds:
            for who in ("self", "other"):
                for condition_id in plan.nuisance_conditions:
                    condition = NUISANCE_LIBRARY[condition_id]
                    identity = stable_id(
                        "nuisance",
                        {
                            "scenario_id": scenario_id,
                            "split": split,
                            "seed": seed,
                            "who": who,
                            "condition": condition_id,
                        },
                    )
                    rows.append(
                        {
                            "trial_id": identity,
                            "scenario_id": scenario_id,
                            "split": split,
                            "seed": seed,
                            "who": who,
                            **asdict(condition),
                        }
                    )
    return tuple(rows)


def validate_split_isolation(rows: Sequence[Mapping[str, Any]]) -> None:
    """Fail if one latent scenario family appears in multiple partitions."""

    by_scenario: dict[str, set[str]] = {}
    for row in rows:
        by_scenario.setdefault(str(row["scenario_id"]), set()).add(str(row["split"]))
    leaking = {key: sorted(value) for key, value in by_scenario.items() if len(value) != 1}
    if leaking:
        raise ValueError(f"scenario-family split leakage: {leaking}")
