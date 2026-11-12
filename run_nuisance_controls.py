"""Run the M3 nuisance-control acquisition against a completed core run."""

from __future__ import annotations

import argparse
from pathlib import Path

from subtext.nuisance_runner import run_nuisance_controls


ROOT = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("core_run", type=Path)
    parser.add_argument(
        "--core-config",
        type=Path,
        default=ROOT / "research/negative-valence/configs/m3_core_v0_3.yaml",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "research/negative-valence/configs/m3_observational_v0_3.yaml",
    )
    arguments = parser.parse_args()
    run_nuisance_controls(
        core_run=arguments.core_run,
        core_config_path=arguments.core_config,
        plan_path=arguments.plan,
    )


if __name__ == "__main__":
    main()
