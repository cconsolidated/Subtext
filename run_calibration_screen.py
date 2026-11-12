"""Run the neutral calibration-only task-family screen."""

from __future__ import annotations

import argparse
from pathlib import Path

from subtext.calibration_screen import run_calibration_screen


DEFAULT_CONFIG = (
    Path(__file__).parent
    / "research"
    / "negative-valence"
    / "configs"
    / "calibration_screen_v0_3.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    run_calibration_screen(arguments.config)


if __name__ == "__main__":
    main()
