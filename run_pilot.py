"""Run the reproducible negative-valence M2 pilot and build its report."""

from __future__ import annotations

import argparse
from pathlib import Path

from subtext.pilot_runner import run_pilot


DEFAULT_CONFIG = Path(__file__).parent / "research" / "negative-valence" / "configs" / "pilot_m2_v0_2.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="run only the first N materialized episodes (integration smoke testing only)",
    )
    arguments = parser.parse_args()
    run_pilot(arguments.config, limit=arguments.limit)


if __name__ == "__main__":
    main()
