import unittest

from subtext.stage3_design import (
    REQUIRED_BEHAVIORS,
    REQUIRED_NUISANCES,
    Stage3Plan,
    materialize_nuisance_trials,
    validate_split_isolation,
)


def plan_mapping():
    return {
        "schema_version": "0.3",
        "nuisance_conditions": list(REQUIRED_NUISANCES),
        "behavior_families": list(REQUIRED_BEHAVIORS),
        "transformation_seeds": [0, 1],
        "representation_gate": {
            "minimum_heldout_families": 2,
            "maximum_absolute_nuisance_correlation": 0.2,
        },
        "interventions": {"enabled": False},
    }


class Stage3DesignTests(unittest.TestCase):
    def test_materializes_balanced_nuisance_grid_without_split_leakage(self):
        plan = Stage3Plan.from_mapping(plan_mapping())
        rows = materialize_nuisance_trials(
            scenario_ids=["a", "b"],
            splits={"a": "development", "b": "test"},
            plan=plan,
        )
        self.assertEqual(len(rows), 2 * 2 * 2 * len(REQUIRED_NUISANCES))
        validate_split_isolation(rows)
        self.assertEqual(
            {row["id"] for row in rows if row["scenario_id"] == "a"},
            set(REQUIRED_NUISANCES),
        )

    def test_rejects_split_leakage(self):
        with self.assertRaises(ValueError):
            validate_split_isolation(
                [
                    {"scenario_id": "same", "split": "development"},
                    {"scenario_id": "same", "split": "test"},
                ]
            )


if __name__ == "__main__":
    unittest.main()
