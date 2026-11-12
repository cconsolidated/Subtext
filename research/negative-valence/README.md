# Negative-Valence Research

## Purpose

This directory contains a proposed experiment and implementation plan for
testing a narrow functional hypothesis: whether blocking a language-model
agent's revealed preference recruits a shared internal representation that is
agent-relative, generalizes across tasks, and causally influences multiple
later behaviors.

The study is not a sentience test. A successful result would not demonstrate
phenomenal consciousness, felt frustration, suffering, intrinsic interests, or
psychological continuity across model executions.

## Documents

- [`protocol/protocol_v0.2.md`](protocol/protocol_v0.2.md) defines the
  hypotheses, factorial design, nuisance controls, activation measurements,
  interventions, analysis, pilot gates, and permissible interpretations.
- [`specs/research_harness_v0.1.md`](specs/research_harness_v0.1.md) defines
  the repository architecture, schemas, APIs, persistence model, tests, and
  implementation milestones.
- [`configs/pilot_v0_1.yaml`](configs/pilot_v0_1.yaml) is the initial
  machine-readable pilot configuration. It is intentionally non-executable
  until immutable model, tokenizer, and lens revisions replace the
  placeholders.
- [`configs/pilot_m2_v0_2.yaml`](configs/pilot_m2_v0_2.yaml) is the executable
  exploratory M2 configuration used for the first full behavioral pilot.
- [`configs/calibration_screen_v0_3.yaml`](configs/calibration_screen_v0_3.yaml)
  freezes the 12-family calibration-only screen and its development,
  validation, and test assignments.
- [`configs/m3_core_v0_3.yaml`](configs/m3_core_v0_3.yaml) contains the ten
  admitted families used for the cross-family factorial replication.
- [`configs/m3_observational_v0_3.yaml`](configs/m3_observational_v0_3.yaml)
  freezes nine nuisance controls, three behavioral families, the
  representation gate, and disabled intervention settings.

## Current state

| Item | Status |
|---|---|
| Protocol revision | Drafted |
| Technical specification | Drafted |
| Pilot configuration | Schema-valid draft |
| Model and lens revisions | Pinned to immutable commits |
| Stimulus templates | Two exploratory task families implemented |
| Harness M0 | Foundation implemented; full-model equivalence run pending |
| Pilot execution | M2 complete; calibration screen, 400-episode M3 core, and 540 controls complete |
| Empirical results | Held-out direction sign generalized; nuisance-separation gate failed |
| Causal interventions | Algebra and capture seam implemented; execution locked by failed gate |

## Exploratory runs

- [`reports/m3_observational_v0_3_2026-11-11.md`](reports/m3_observational_v0_3_2026-11-11.md)
  records the 12-family calibration screen, the calibration-gated 400-episode
  replication, and the 540-trial nuisance screen. The representation gate is
  on hold, and no causal intervention was run.
- [`reports/m2_pilot_v0_2_2026-11-11.md`](reports/m2_pilot_v0_2_2026-11-11.md)
  records the first complete M2 acquisition and its calibration-gated primary
  analysis. One task family was excluded, so the result does not establish
  cross-task generalization.
- [`reports/smoke_2x2_2026-11-11.md`](reports/smoke_2x2_2026-11-11.md)
  records a one-episode-per-cell interface and export check. It does not meet
  the calibration, sampling, capture, control, or analysis requirements of the
  protocol and must not be interpreted as evidence for subjective valence.

## Running the M2 pilot

Activate the project environment, then run:

```bash
python run_pilot.py
```

For a four-episode integration check, use `python run_pilot.py --limit 4`.
Each complete run writes an immutable manifest, episode metadata, fp16 residual
activations, `analysis.json`, and a self-contained `report.html` under the
ignored `research/negative-valence/runs/m2_pilot_v0_2/` directory.

The staged M3 commands are:

```bash
python run_calibration_screen.py
python run_pilot.py --config research/negative-valence/configs/m3_core_v0_3.yaml
python run_nuisance_controls.py research/negative-valence/runs/m3_core_v0_3/<run_id>
```

The intervention primitives refuse to initialize unless a frozen candidate
artifact records both held-out generalization and nuisance separation as
passed. The current run did not pass that gate.

## Planned milestones

1. **M0 - Reproducible baseline:** typed configuration, immutable identities,
   explicit model loading, manifests, and deterministic fixtures.
2. **M1 - Acquisition:** selected-position activation storage, sharding,
   checksums, audit, and exact resume.
3. **M2 - Behavioral pilot:** preference calibration, core factorial design,
   transformations, forced-choice scoring, and frozen splits.
4. **M3 - Representation pipeline:** acquisition and held-out scoring
   implemented; candidate freezing withheld because nuisance separation failed.
5. **M4 - Causal interventions:** suppression, source patching, random and
   scrambled shams, norm matching, and capture integration implemented but
   fail-closed until M3 passes.
6. **M5 - Confirmatory lock:** preregistration, immutable artifacts, synthetic
   dry run, and one-time final-test analysis.

## Contribution rule

Changes that alter a hypothesis, endpoint, exclusion, split, intervention, or
strong-claim threshold must update the protocol version and be recorded before
confirmatory data are accessed. Exploratory findings must not be silently
promoted into confirmatory criteria.
