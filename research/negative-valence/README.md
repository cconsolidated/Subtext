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

## Current state

| Item | Status |
|---|---|
| Protocol revision | Drafted |
| Technical specification | Drafted |
| Pilot configuration | Schema-valid draft |
| Model and lens revisions | Pinned to immutable commits |
| Stimulus templates | Two exploratory task families implemented |
| Harness M0 | Foundation implemented; full-model equivalence run pending |
| Pilot execution | M2 harness implemented; 80-episode exploratory pilot completed |
| Empirical results | One of two task families passed calibration; cross-task gate not met |

## Exploratory runs

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

## Planned milestones

1. **M0 - Reproducible baseline:** typed configuration, immutable identities,
   explicit model loading, manifests, and deterministic fixtures.
2. **M1 - Acquisition:** selected-position activation storage, sharding,
   checksums, audit, and exact resume.
3. **M2 - Behavioral pilot:** preference calibration, core factorial design,
   transformations, forced-choice scoring, and frozen splits.
4. **M3 - Representation pipeline:** nuisance probes, candidate selection,
   candidate freezing, and pilot power analysis.
5. **M4 - Causal interventions:** suppression, source patching, matched shams,
   dose schedules, and competence controls.
6. **M5 - Confirmatory lock:** preregistration, immutable artifacts, synthetic
   dry run, and one-time final-test analysis.

## Contribution rule

Changes that alter a hypothesis, endpoint, exclusion, split, intervention, or
strong-claim threshold must update the protocol version and be recorded before
confirmatory data are accessed. Exploratory findings must not be silently
promoted into confirmatory criteria.
