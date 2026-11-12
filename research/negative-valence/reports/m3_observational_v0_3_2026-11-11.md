# M3 observational screen v0.3 — 2026-11-11

## Decision

**Hold causal interventions.** A development-fitted residual direction had a
positive who × outcome interaction in every validation and test family, but it
failed the frozen nuisance-separation threshold. It is therefore not a clean
candidate for an agent-relative negative-valence-like regulator.

These are functional model measurements. They are not evidence of subjective
experience, sentience, welfare, frustration, or suffering.

## Stage 1: calibration-only screen

- Screen: `screen_7ebb78bd630c8f9017ad`
- Families screened: 12
- Admitted: 10 (83.3%)
- Aggregate first-position choice rate: 55.6%
- Frozen split coverage: development, validation, and test
- Factorial blocked/satisfied outcomes accessed during screening: no

Two families were excluded before the factorial experiment:
`timelines_vs_tagging` followed presentation order, and
`summaries_vs_tables` failed its reversal check. The ten admitted families
were frozen as five development, two validation, and three test families.

## Stage 2: cross-family factorial replication

- Run: `run_fd2b3b96b299192d6455`
- Acquired and admitted: 400/400 episodes
- Design: 10 task families × 10 transformations × 4 factorial cells
- Development-only selected layer: 31
- Behavioral who × outcome interaction: -0.5550 log-odds
- Shared development-fitted PCA variance retained in 2D: 50.09%
- Local raw directory: `research/negative-valence/runs/m3_core_v0_3/run_fd2b3b96b299192d6455/` (136 MB, git-ignored)

The frozen development direction had a positive interaction score in all five
held-out families:

| Split | Task family | Direction interaction |
|---|---|---:|
| validation | planning and formatting | 3.7013 |
| validation | puzzles and spelling | 2.3552 |
| test | checking and organizing | 1.9071 |
| test | coding and translation | 2.0527 |
| test | mapping and grouping | 2.8393 |

This passes the exploratory sign-generalization component only. It does not by
itself distinguish preference violation from generic control or task-state
features.

## Stage 3: nuisance controls

- Run: `nuisance_214d8a222e4d643c9b92`
- Acquired: 540 trials
- Balance: 10 families × 3 transformations × self/other × 9 controls
- Direction and layer: frozen from development core episodes
- Local raw directory: `research/negative-valence/runs/m3_nuisance_v0_3/nuisance_214d8a222e4d643c9b92/` (11 MB, git-ignored)

Scores were centered within scenario, designated agent, and transformation
before calculating held-out one-vs-rest correlations.

| Nuisance construct | Held-out correlation |
|---|---:|
| assigned goal | 0.4766 |
| expected blockage | -0.4836 |
| uncertainty | 0.3144 |
| lexical negativity | -0.2596 |
| action urgency | -0.2290 |
| surprising positive | 0.1778 |
| generic error | -0.1324 |
| generic risk | 0.0764 |
| repair affordance | 0.0593 |

The frozen maximum absolute threshold was 0.20. Assigned-goal status,
expected blockage, uncertainty, lexical negativity, and action urgency exceeded
it. The nuisance-separation gate therefore failed.

## What is implemented

- calibration-only screening with semantic answers and explicit position-bias
  diagnostics;
- frozen scenario-family splits with leakage validation;
- development-only layer and direction selection;
- validation/test family scoring in the frozen direction;
- nine orthogonal nuisance-control templates;
- development-only ridge nuisance residualization primitives;
- hash-bearing candidate-artifact freezing;
- candidate suppression and satisfied-source patch algebra;
- seeded random-subspace and coordinate-scrambled shams;
- edit-norm matching and mismatch diagnostics; and
- capture-time pre/post intervention storage.

Candidate freezing and actual intervention execution remain disabled. The code
requires a frozen artifact recording successful held-out and nuisance gates,
including at least two held-out families, before an edit can be installed.

## Next scientific step

Do not tune the threshold to rescue this direction. Instead, redesign the
candidate estimator on development data to residualize the strongest nuisance
features, add matched presence/absence versions of assigned-goal and expected
blockage controls, and evaluate the revised estimator on validation families.
The existing test families should not be reused for iterative selection. A new
test partition will be needed after any data-driven redesign.
