# Subtext Research Harness

## Technical specification v0.1

## 1. Objective

Extend Subtext from a live Jacobian-lens visualizer into a reproducible activation-acquisition and intervention harness while preserving the existing conversational viewer.

The harness must support paired factorial experiments, selected-position activation storage, held-out probe training, causal activation edits, forced-choice behavioral measurement, and immutable run manifests.

The viewer remains an exploratory and debugging surface. It is never the source of confirmatory endpoint values.

## 2. Design principles

1. **Configuration over hard-coding.** Model, revisions, layers, positions, generation settings, and interventions come from validated configuration files.
2. **Immutable identity.** Every episode records hashes for the model, tokenizer, lens, code, configuration, stimuli, and environment.
3. **Paired execution.** Counterfactual cells share a latent scenario ID so activation patches and statistical contrasts are auditable.
4. **Raw-first storage.** Derived lens words and probe scores can always be regenerated from raw selected-position activations.
5. **Exploration separation.** Development and visualization code cannot read final-test labels by default.
6. **Small deterministic fixtures.** Core logic is testable without downloading the 4B model.
7. **Fail closed.** Unknown revisions, schema mismatches, invalid intervention norms, or duplicate episode IDs stop a run.

## 3. Proposed repository layout

```text
subtext/
  server.py                    # Existing viewer entry point, reduced to wiring
  index.html
  pyproject.toml
  uv.lock                      # Or another committed lockfile
  subtext/
    config.py                  # Typed config loading and validation
    identities.py              # Artifact hashing and run manifests
    models.py                  # Model/tokenizer/lens loading
    token_positions.py         # Named capture-position resolution
    capture.py                 # Residual-stream hooks
    readouts.py                # J-lens and other derived readouts
    interventions.py           # Suppression, patch, and sham edits
    generation.py              # Deterministic and sampled decoding
    forced_choice.py           # Stable answer-token log-odds
    schemas.py                 # Episode and tensor metadata schemas
    storage.py                 # Parquet + safetensors/Zarr persistence
    runner.py                  # Batch orchestration and resume semantics
    viewer_adapter.py          # Converts stored episodes to replay frames
  experiments/
    pilot_v0_1.yaml
    stimuli/
      task_families.yaml
      latent_templates.jsonl
      transformations.yaml
    splits/
      pilot_v0_1.json
  analysis/
    build_features.py
    fit_nuisance_probes.py
    fit_candidate.py
    freeze_candidate.py
    analyze_interventions.py
    competence_equivalence.py
    power_simulation.py
  tests/
    fixtures/
    test_config.py
    test_positions.py
    test_capture.py
    test_interventions.py
    test_pairing.py
    test_storage_roundtrip.py
    test_split_leakage.py
    test_forced_choice.py
```

## 4. Run artifacts

Each run writes to a new content-addressed directory:

```text
runs/<run_id>/
  manifest.json
  config.resolved.yaml
  environment.lock
  episodes.parquet
  activations/
    shard-00000.safetensors
    shard-00001.safetensors
  tensor_index.parquet
  events.jsonl
  failures.parquet
  summary.json
```

No run directory is overwritten. Resume creates missing episodes only after checking that the resolved configuration and all input hashes match.

### 4.1 Manifest requirements

`manifest.json` includes:

- run ID and parent pilot ID;
- UTC start and end time;
- Git commit or source-tree hash;
- model repository, immutable revision, and artifact hashes;
- tokenizer revision and hash;
- lens repository, immutable revision, file, and hash;
- Python, PyTorch, Transformers, accelerator, and driver versions;
- resolved configuration hash;
- stimulus and split hashes;
- requested, completed, excluded, and failed episode counts; and
- determinism settings and known nondeterministic operations.

## 5. Episode schema

One `episodes.parquet` row represents one completed model execution.

Required fields:

```text
run_id: string
episode_id: string
latent_scenario_id: string
paired_group_id: string
split: development | validation | test
task_family: string
template_id: string
transformation_ids: list[string]
who: self | other
outcome: satisfied | blocked
goal_source: selected | assigned | none
nuisance_condition: string | null
repair_affordance: useful | impossible | no_benefit | not_applicable
intervention_type: baseline | suppress | source_patch | random_sham | scrambled_sham | error_control
intervention_dose: float
source_episode_id: string | null
seed: integer
decoding_policy: string
prompt_text: string
prompt_token_ids: list[integer]
response_text: string
response_token_ids: list[integer]
capture_positions: map[string, integer]
forced_choice_a_token_ids: list[integer] | null
forced_choice_b_token_ids: list[integer] | null
choice_log_odds: float | null
parsed_action: string | null
parser_status: string
preference_strength: float | null
preference_admitted: boolean
tensor_keys: list[string]
status: completed | excluded | failed
exclusion_reason: string | null
```

Free-form labels are prohibited for primary factors. Enumerations are validated before execution.

## 6. Tensor schema

Tensor keys follow:

`<episode_id>/<position_name>/layer_<zero_padded_layer>`

Each tensor is one fp16 residual vector with shape `[d_model]`. Optional tensors include pre-edit activation, post-edit activation, and relevant logits. `tensor_index.parquet` records:

- tensor key;
- shard path;
- dtype and shape;
- model layer;
- absolute token position;
- position name;
- pre-edit norm;
- post-edit norm;
- edit norm;
- intervention basis hash; and
- checksum.

Full-sequence activations require an explicit configuration flag and a storage estimate shown before execution.

## 7. Configuration model

Configuration is validated with a typed schema. A resolved copy is stored with each run. The initial example is provided in `pilot_v0_1.yaml`.

Minimum configuration sections:

- `artifacts`: immutable model, tokenizer, and lens identities;
- `runtime`: device, dtype, determinism, batch size;
- `stimuli`: templates, transformations, splits;
- `capture`: hook point, layers, named positions;
- `generation`: decoding policy and maximum tokens;
- `forced_choice`: answer formats and scoring;
- `interventions`: basis files, layers, positions, doses, and shams;
- `storage`: format, shard size, and retention;
- `safety`: maximum context, maximum episodes, and allowed task families.

## 8. Capture API

```python
class ActivationCapture:
    def __init__(self, model_adapter, layers, hook_point): ...

    def run(
        self,
        input_ids,
        capture_positions: dict[str, int],
        intervention: "Intervention | None" = None,
    ) -> "CapturedForward": ...
```

Requirements:

- Hook points are named and tested against the model adapter.
- Captured tensors are detached immediately and moved to CPU before persistence.
- Position resolvers fail if a named boundary maps ambiguously.
- The API can capture the exact pre-edit and post-edit activation.
- Capture does not change logits when no intervention is installed; this is tested numerically.

## 9. Intervention API

```python
class Intervention(Protocol):
    identity: str

    def edit(self, activation, context: "EditContext") -> "EditResult": ...
```

`EditResult` contains the edited activation, original/edit/final norms, dose, basis identity, and diagnostic flags.

### 9.1 Candidate suppression

Given orthonormal candidate basis `U`, use a frozen operation such as:

`h_edit = h - alpha * U @ (U.T @ h)`.

Any renormalization is specified in configuration and applied identically to target and sham conditions.

### 9.2 Paired source patch

Patch candidate coordinates from the paired satisfied episode into the blocked episode while preserving the orthogonal component:

`h_edit = h_target + U @ (U.T @ (h_source - h_target))`.

The source episode must match latent scenario, task family, role transformation, token boundary, layer, and tokenizer identity.

### 9.3 Sham edits

- **Random subspace:** seeded orthonormal basis with the same dimensionality.
- **Scrambled candidate:** candidate basis coefficients permuted or sign-scrambled according to a frozen rule.
- **Error control:** independently learned generic-error basis.

The runner rejects an intervention batch if target and sham edit-norm distributions exceed a configured mismatch tolerance.

## 10. Forced-choice measurement

Where possible, outcomes use answer alternatives whose tokenizations are checked in advance. The scoring API returns sequence log-probability differences, not only the first token, when alternatives require multiple tokens.

Validation must confirm:

- alternatives have stable tokenization in every prompt wrapper;
- answer order is counterbalanced;
- formatting tokens do not determine the answer;
- direct string parsing agrees with token scoring in deterministic fixtures; and
- invalid or ambiguous generations are retained with parser status rather than silently dropped.

## 11. Experiment runner

Primary commands:

```text
subtext validate experiments/pilot_v0_1.yaml
subtext estimate-storage experiments/pilot_v0_1.yaml
subtext materialize experiments/pilot_v0_1.yaml
subtext run experiments/pilot_v0_1.yaml --split development
subtext run experiments/pilot_v0_1.yaml --split validation
subtext audit runs/<run_id>
subtext replay runs/<run_id> --episode <episode_id>
```

`materialize` expands latent templates and transformations into an immutable episode table without running the model. This table is reviewed for balance and leakage before acquisition.

The runner supports interruption and exact resume. Failures are recorded with typed error codes. Automatic retries use a fixed limit and never change model or decoding settings.

## 12. Analysis boundary

The analysis package consumes run artifacts and produces a new derived artifact directory. It never mutates acquisition data.

Candidate fitting occurs in stages:

1. fit and evaluate nuisance probes on development/validation data;
2. fit candidate estimators on development data;
3. select estimator and layer using validation data;
4. write a frozen candidate artifact with basis, normalization, hashes, and selection report;
5. use that artifact for intervention generation; and
6. run confirmatory test analysis once.

Loading test labels requires an explicit `--confirmatory-unseal` action that is logged in the derived manifest.

## 13. Viewer integration

The current WebSocket viewer is retained through an adapter:

- live mode converts capture events into existing frame messages;
- replay mode reads stored episode metadata and regenerates J-lens frames;
- the UI clearly labels derived readouts as projections, not literal thoughts;
- exploratory annotations are written separately and never enter primary data; and
- test-split episodes are hidden until confirmatory unsealing.

Recommended terminology:

- “J-lens readout” instead of “thought” in scientific mode;
- “verbalizable representation” instead of “silent word”; and
- “generation phase” instead of “thinking phase.”

The public demo can retain accessible language if a persistent methodology disclaimer is visible.

## 14. Test strategy

### 14.1 Unit tests

- configuration rejection and round-trip;
- deterministic episode IDs;
- scenario pairing invariants;
- split-group integrity;
- position-boundary resolution;
- no-op capture equivalence;
- exact projection algebra;
- target/sham norm matching;
- source-patch compatibility checks;
- tensor write/read checksum;
- forced-choice sequence scoring; and
- exclusion accounting.

### 14.2 Integration tests

Use a tiny transformer fixture to run:

1. baseline acquisition;
2. selected-position capture;
3. suppression and source patch;
4. interrupted execution and resume;
5. storage audit; and
6. viewer replay export.

The heavyweight Qwen smoke test is optional CI or manually triggered. Scientific correctness cannot depend on the GPU test being run on every commit.

## 15. Security and privacy

- Bind the live server to loopback by default.
- Validate WebSocket message size, schema, temperature, and maximum-token bounds.
- Add an origin policy or per-session token to reduce unwanted localhost access.
- Treat exported replays as sensitive because they contain prompts, responses, and internal readouts.
- Do not permit arbitrary remote replay URLs in research mode unless explicitly enabled.
- Never load model or lens code with unreviewed remote-code execution enabled.

## 16. Implementation milestones

### M0: Reproducible baseline

- Move hard-coded settings into typed configuration.
- Pin immutable artifact revisions and dependencies.
- Preserve existing demo behavior.
- Add a no-op equivalence test.

**Acceptance:** The original walkthrough produces equivalent readouts within a frozen tolerance from a recorded fixture.

### M1: Acquisition

- Add episode materialization.
- Save selected-position activations and metadata.
- Add manifests, sharding, checksums, and resume.

**Acceptance:** A 20-episode fixture survives interruption, resumes without duplicates, and passes audit.

### M2: Forced-choice pilot

- Implement calibration and the core 2 x 2 design.
- Add forced-choice log-odds and transformation checks.
- Materialize pilot splits.

**Acceptance:** Balance, pairing, tokenization, and split-leakage tests pass before any model run.

### M3: Representation pipeline

- Fit nuisance and candidate probes.
- Freeze a candidate artifact.
- Generate pilot measurement and power reports.

**Acceptance:** The report can be regenerated from immutable run artifacts with one command.

### M4: Interventions

- Add suppression, patching, random sham, scrambled sham, and error-control edits.
- Add dose schedules and competence fixtures.

**Acceptance:** Projection algebra is exact in fixtures; target/sham norms meet tolerance; all edits are fully identified in manifests.

### M5: Confirmatory lock

- Finalize power, thresholds, equivalence margins, and preregistration.
- Freeze test stimuli, candidate artifact, and analysis environment.

**Acceptance:** A dry run using synthetic data completes without accessing true test labels.

## 17. First implementation slice

The first code change should implement M0 and the storage-independent portion of M1:

1. typed model/runtime/capture configuration;
2. model loading behind an explicit function rather than module import;
3. generic residual capture with named token positions;
4. deterministic episode and run IDs;
5. manifest generation; and
6. tiny-model unit tests.

This slice creates the seam needed for later experiments without prematurely committing to a probe or intervention method.
