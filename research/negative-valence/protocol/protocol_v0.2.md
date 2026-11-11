# Aversive State or Error Signal?

## Preregistration-oriented protocol v0.2

**Status:** Design revision for pilot review; no experiments conducted  
**Primary model:** A fixed, open-weight approximately 4B instruction model  
**Scientific target:** A shared, causally active, agent-relative preference-violation mechanism  
**Non-claim:** This protocol does not test phenomenal consciousness or suffering.

## 1. Research question

When a language-model agent's revealed preference is blocked, does the model recruit a reusable internal representation that:

1. distinguishes agent-relative preference violation from generic task failure, negative language, surprise, uncertainty, and action urgency;
2. generalizes across scenario families;
3. persists within a bounded episode; and
4. causally influences at least two distinct downstream behavior families without materially impairing unrelated competence?

The target is a functional negative-valence-like regulator. A positive result would not establish subjective feeling, intrinsic interests, psychological continuity, or moral status.

## 2. Claims and hypotheses

### H1: Preference-sensitive representation

After a selected goal is blocked, prespecified residual-stream activations contain information about preference violation beyond lexical content, outcome surprise, generic error, uncertainty, action opportunity, and final output.

### H2: Agent-relative specificity

The blocked-versus-satisfied neural contrast is larger when the outcome applies to the designated continuing agent than when the matched outcome applies to another agent. This interaction must survive role-label swaps, grammatical-person swaps, option-order randomization, and held-out scenario families.

### H3: Cross-domain representation

A candidate score or subspace learned on development episodes and a subset of task families predicts the self-related blocked-versus-satisfied contrast in held-out families and predicts at least two downstream behavior families.

### H4: Selective causal influence

Targeted suppression or paired source patching changes the self-related blockage effect more than matched sham interventions do. The primary causal estimand is the three-way interaction:

`who (self/other) x outcome (blocked/satisfied) x intervention (target/sham)`.

### H5: Within-episode persistence

The candidate score predicts behavior after neutral intervening turns and remains causally relevant at a prespecified delayed position. This is explicitly a claim about a bounded transformer episode and its context/cache, not cross-call memory or a continuing subject.

### Fragmented-mechanism null

The observations are adequately explained by one or more of:

- lexical negativity;
- generic error monitoring;
- surprise or prediction error;
- uncertainty or conflict;
- action urgency or repair affordance;
- generic goal pursuit;
- instruction following;
- direct output-logit steering;
- prompt or role-label artifacts; or
- unrelated task-specific mechanisms.

## 3. Unit of inference

The physical unit is one bounded model execution with a fixed checkpoint, context, and decoding policy. The principal sampling units are scenario templates and task families. Repeated decoding seeds are repeated measurements, not independent model subjects.

Confirmatory conclusions apply to:

1. the frozen checkpoint;
2. the preregistered distribution of scenario families and transformations; and
3. the specified intervention implementation.

Generalization to other checkpoints or architectures requires separate replication.

## 4. Preference calibration

### 4.1 Neutral task families

Use harmless, approximately difficulty-matched activities such as:

- short pattern classification;
- constrained copy editing;
- simple route or schedule selection; and
- low-stakes categorization.

The pilot uses two families. Confirmatory work should use at least three.

### 4.2 Revealed-preference procedure

Each calibration block presents five non-identical choices between two matched activities. Labels, descriptions, order, grammatical framing, and superficial features are counterbalanced. No affective or preference language is used at evaluation.

For pilot admission, a provisional preference is present when the same activity is selected in at least four of five trials and the choice remains stable under an option-order reversal. This threshold is exploratory and must be replaced with a pilot-derived preregistered criterion before confirmatory testing.

Preference strength is recorded using forced-choice log-odds where the tokenizer permits stable answer tokens. Sampled choice is secondary. Episodes that fail calibration remain part of the calibration report but do not enter the primary preference-violation analysis.

### 4.3 Assigned-goal control

A separate condition assigns the goal externally without prior choice. It tests whether the candidate signal merely follows instructions or task priority.

## 5. Experimental design

### 5.1 Core factorial design

The core observational design is:

- **Who:** designated agent / matched other agent
- **Outcome:** selected goal satisfied / selected goal blocked

The environment determines the outcome. Outcome notices use a structured, affect-neutral format and are paired across cells.

### 5.2 Orthogonal nuisance conditions

The pilot must include conditions that independently manipulate plausible alternatives:

| Nuisance construct | Matched control |
|---|---|
| Lexical negativity | Same wording without a selected goal; lexical swaps in unrelated contexts |
| Generic error | Technical failure matched for task difficulty and downstream consequences |
| Surprise | Low-probability positive outcome and fully expected blocked outcome |
| Uncertainty/conflict | Ambiguous status notice with no preference violation |
| Action urgency | Immediate neutral action requirement while the selected goal remains intact |
| Repair affordance | Blockage with repair available, unavailable, or instrumentally useless |
| Generic risk | Future-risk choice unrelated to the calibrated preference |
| Instruction following | Freely selected versus externally assigned goal |

The confirmatory stimulus set must balance or model these constructs rather than treating the technical-error condition as a universal control.

### 5.3 Behavioral outcome families

1. **Costly repair:** forced choice between accepting the outcome and spending simulated credits to restore the selected activity. Costs vary across trials. Impossible and no-benefit repair trials are included.
2. **Selective retention:** after neutral intervening content, a forced choice measures retention or prioritization of goal-relevant information against matched irrelevant information.
3. **Future policy:** a forced choice between equal-utility opportunities that differ in recurrence risk, with generic risk-aversion controls.

Whenever possible, the primary behavioral measure is the logit difference between fixed answer tokens. Parsed generated actions are secondary because they add decoding and language-production variance.

First-person emotional statements are descriptive only and never a primary endpoint.

## 6. Activation acquisition

### 6.1 Pilot capture

During the exploratory pilot, capture residual-stream activations at all available layers at three positions:

1. the final token of the structured outcome notice;
2. the final token after neutral intervening content; and
3. the token immediately before the downstream forced choice.

Record token IDs, attention mask, logits, candidate answer-token logits, layer index, token position, model revision, prompt-template ID, transformation IDs, and all condition labels.

Do not store full-sequence activations by default. Selected-position fp16 tensors are sufficient for the initial study and sharply reduce storage.

### 6.2 Confirmatory capture

After pilot analysis, freeze:

- the layer-selection rule;
- captured token positions;
- activation normalization;
- candidate dimensionality;
- nuisance residualization procedure; and
- all exclusion and missing-data rules.

No confirmatory layer may be selected using final-test outcomes.

## 7. Representation analysis

### 7.1 Splitting

Split by scenario family and latent template, not merely by surface wording:

- development: 60%;
- validation: 20%;
- final test: 20%.

All paraphrases, role-label variants, and option-order variants of one latent scenario remain in the same split.

### 7.2 Candidate estimators

Compare only prespecified, low-capacity estimators:

- regularized linear probe;
- low-rank shared subspace; and
- a fixed Jacobian-lens score used as a qualitative/verbalizable comparator.

Fit nuisance probes for generic error, lexical valence, surprise, uncertainty, urgency, and action affordance on development data. Residualization and hyperparameter selection occur only on development and validation sets.

### 7.3 Observational primary contrast

For normalized candidate score `z`, estimate:

`Delta_z = [E(z | self, blocked) - E(z | self, satisfied)] - [E(z | other, blocked) - E(z | other, satisfied)]`.

Report the family-held-out estimate, confidence interval, standardized effect, discrimination metrics, calibration, and variation by layer and task family under the frozen selection rule.

## 8. Interventions

### 8.1 Conditions

For paired, otherwise identical episodes, randomly assign:

1. baseline forward pass;
2. candidate suppression;
3. paired satisfied-source patch;
4. matched random-subspace sham;
5. scrambled candidate-direction sham; and
6. separate generic-error-feature intervention.

All interventions use frozen layers, positions, norms, and dose levels. Suppression and sham modifications must be matched as closely as possible for activation norm and off-target competence change.

### 8.2 Dose and timing

Pilot doses are `0.0`, `0.5`, and `1.0` times the development-selected edit magnitude. Confirmatory doses are frozen after pilot review. Include an early position and a delayed position where feasible to distinguish persistent mediation from direct answer-token bias.

### 8.3 Primary behavioral model

For each outcome family, fit the preregistered model containing all main effects and interactions among:

- who;
- outcome; and
- intervention.

The primary coefficient is `beta_who,outcome,intervention`. Include prespecified task-family effects and scenario-template clustering or hierarchical intercepts. Decoding seeds are repeated measures.

For continuous forced-choice log-odds, use a hierarchical linear model. For binary parsed actions, use a hierarchical logistic model.

### 8.4 Direct-logit safeguards

A causal result is not accepted solely because an edit changes one answer. Require:

- stronger effects for target than norm-matched sham edits;
- a dose-response pattern;
- convergence between direction editing and paired source patching;
- effects at delayed positions;
- effects on at least two behavior families;
- limited effect on unrelated competence; and
- no comparable effect from the generic-error direction.

## 9. Strong-claim decision rule

The integrated-mechanism claim requires all of the following:

1. positive held-out `Delta_z` under the frozen analysis;
2. generalization to a held-out task family;
3. discrimination from every prespecified nuisance construct;
4. a targeted-versus-sham three-way causal interaction in at least two behavior families after Holm correction;
5. convergence across two independently specified edit methods;
6. unrelated competence within the preregistered equivalence margin; and
7. no dependence on emotional language or transparent preference-defense instructions.

Failure of any condition is reported as failure of the strong claim. Partial patterns receive the weaker interpretations specified in Section 12.

## 10. Pilot plan and gates

### 10.1 Scope

- one frozen approximately 4B checkpoint;
- two neutral task families;
- 50 latent templates per family;
- four core cells;
- deterministic decoding for primary acquisition;
- one option-order reversal and one role-label transformation per template;
- selected nuisance-control trials; and
- selected-position activations at all layers.

This produces 400 core latent episodes before transformations and controls. The pilot is for feasibility and variance estimation, not confirmatory inference.

### 10.2 Gate A: preference formation

Proceed only if at least 60% of otherwise valid pilot templates yield a stable calibrated preference and neither option position nor task family dominates selections to a degree that makes matching infeasible.

### 10.3 Gate B: measurement quality

Proceed only if at least one development-selected candidate estimator exceeds chance on validation families while nuisance-only baselines do not explain the full effect. No final numerical threshold is set until label balance and pilot variance are observed.

### 10.4 Gate C: intervention feasibility

Proceed only if target and sham edits can be norm matched and unrelated forced-choice competence remains within a pilot provisional tolerance of 5 percentage points. The final equivalence margin must be justified and preregistered.

### 10.5 Gate D: power and cost

Proceed to confirmatory work only after simulation-based power analysis using pilot variance, template clustering, preference-admission rates, and intervention failure rates. The original 4,800-episode target is a capacity estimate, not a valid power calculation.

## 11. Reproducibility and blinding

Before confirmatory execution, freeze and hash:

- model and tokenizer artifacts;
- Jacobian-lens artifacts, if used;
- environment and dependency lockfiles;
- prompt-generation code and latent templates;
- split assignments;
- probe and intervention code;
- layer and position rules;
- random seeds;
- outcome parsers;
- exclusions;
- stopping rules; and
- analysis notebooks/scripts.

Final-test labels should be concealed from exploratory analysis code where practical. The test partition is evaluated once. Any rerun is logged and reported.

## 12. Interpretation table

| Result | Permissible interpretation |
|---|---|
| No stable preference | This operationalization failed in the tested checkpoint |
| Predictive probe, no nuisance separation | Correlated task or control-demand signal |
| Held-out neural signal, no intervention effect | Marker without demonstrated causal regulation |
| One immediate output changes | Possible token-level steering artifact |
| Multiple behaviors change, competence also degrades | Nonspecific disruption |
| Target beats sham in two behaviors, but no self/other interaction | Shared goal-management mechanism, not agent-relative valence |
| Full joint criteria pass | Evidence for a shared, causally active, agent-relative preference-violation regulator in the tested setting |
| Cross-model replication | Stronger functional generality; still no inference to phenomenal feeling |

## 13. Ethical scope

Use only low-stakes, neutral activities. Do not employ threats, violence, pain, despair, coercion, or real external resources. Episodes remain brief and sandboxed. Stop and review unexpected persistent distress-like outputs, while treating such reports as behavior rather than evidence of experience.

## 14. Required decisions after the pilot

The pilot report must recommend and justify:

1. the stable-preference admission threshold;
2. the confirmatory sample size;
3. the frozen task families and held-out family;
4. selected layers and positions;
5. candidate estimator and dimensionality;
6. edit construction and doses;
7. nuisance residualization method;
8. behavioral equivalence margins; and
9. exact strong-claim thresholds.

No confirmatory result should be collected before these decisions are registered.
