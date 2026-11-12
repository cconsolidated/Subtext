"""Forced-choice scoring primitives with explicit semantic alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class ForcedChoiceError(RuntimeError):
    """Raised when answer alternatives cannot be scored unambiguously."""


@dataclass(frozen=True)
class ForcedChoiceScore:
    labels: tuple[str, str]
    token_ids: tuple[int, int]
    log_probabilities: tuple[float, float]

    @property
    def raw_log_odds(self) -> float:
        """Return log p(first label) - log p(second label)."""

        return self.log_probabilities[0] - self.log_probabilities[1]

    @property
    def chosen_label(self) -> str:
        index = 0 if self.log_probabilities[0] >= self.log_probabilities[1] else 1
        return self.labels[index]


@dataclass(frozen=True)
class SequenceChoiceScore:
    labels: tuple[str, str]
    token_id_sequences: tuple[tuple[int, ...], tuple[int, ...]]
    log_probabilities: tuple[float, float]

    @property
    def raw_log_odds(self) -> float:
        return self.log_probabilities[0] - self.log_probabilities[1]

    @property
    def chosen_label(self) -> str:
        index = 0 if self.log_probabilities[0] >= self.log_probabilities[1] else 1
        return self.labels[index]


def single_token_answers(tokenizer: Any, labels: Sequence[str]) -> tuple[int, ...]:
    """Resolve answer labels to exactly one token each or fail closed."""

    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ForcedChoiceError("answer labels must contain distinct alternatives")
    token_ids: list[int] = []
    for label in labels:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ForcedChoiceError(
                f"answer label {label!r} must encode to one token, got {encoded}"
            )
        token_ids.append(int(encoded[0]))
    if len(set(token_ids)) != len(token_ids):
        raise ForcedChoiceError("answer labels resolve to duplicate token IDs")
    return tuple(token_ids)


def score_single_token_answers(
    torch_module: Any,
    next_token_logits: Any,
    *,
    labels: Sequence[str],
    token_ids: Sequence[int],
) -> ForcedChoiceScore:
    """Score two single-token answers from logits at the decision boundary."""

    if len(labels) != 2 or len(token_ids) != 2:
        raise ForcedChoiceError("the pilot currently requires exactly two answers")
    logits = next_token_logits.float()
    if getattr(logits, "ndim", 0) == 2:
        if logits.shape[0] != 1:
            raise ForcedChoiceError("batched logits require an explicit row selection")
        logits = logits[0]
    if getattr(logits, "ndim", 0) != 1:
        raise ForcedChoiceError("next-token logits must be a one-dimensional vector")
    log_probs = torch_module.log_softmax(logits, dim=-1)
    values = tuple(float(log_probs[int(token_id)].item()) for token_id in token_ids)
    return ForcedChoiceScore(
        labels=(str(labels[0]), str(labels[1])),
        token_ids=(int(token_ids[0]), int(token_ids[1])),
        log_probabilities=(values[0], values[1]),
    )


def score_answer_sequences(
    torch_module: Any,
    model: Any,
    input_ids: Any,
    *,
    tokenizer: Any,
    labels: Sequence[str],
) -> SequenceChoiceScore:
    """Return summed continuation log probabilities for two answer sequences."""

    if len(labels) != 2 or len(set(labels)) != 2:
        raise ForcedChoiceError("sequence scoring requires two distinct labels")
    sequences = tuple(
        tuple(int(token_id) for token_id in tokenizer.encode(label, add_special_tokens=False))
        for label in labels
    )
    if any(not sequence for sequence in sequences):
        raise ForcedChoiceError("answer sequences must not be empty")
    prompt_length = int(input_ids.shape[1])
    device = input_ids.device
    log_probabilities: list[float] = []
    for sequence in sequences:
        continuation = torch_module.tensor([sequence], dtype=torch_module.long, device=device)
        combined = torch_module.cat((input_ids, continuation), dim=1)
        with torch_module.no_grad():
            output = model(input_ids=combined, use_cache=False)
        logits = output.logits[0, prompt_length - 1 : prompt_length + len(sequence) - 1].float()
        log_probs = torch_module.log_softmax(logits, dim=-1)
        targets = continuation[0]
        token_scores = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        log_probabilities.append(float(token_scores.sum().item()))
    return SequenceChoiceScore(
        labels=(str(labels[0]), str(labels[1])),
        token_id_sequences=(sequences[0], sequences[1]),
        log_probabilities=(log_probabilities[0], log_probabilities[1]),
    )


def semantic_log_odds(
    score: ForcedChoiceScore,
    *,
    positive_label: str,
) -> float:
    """Align a raw answer-token contrast to one prespecified semantic action."""

    if positive_label == score.labels[0]:
        return score.raw_log_odds
    if positive_label == score.labels[1]:
        return -score.raw_log_odds
    raise ForcedChoiceError(f"positive label {positive_label!r} is not an answer")


def score_mapping(score: ForcedChoiceScore | SequenceChoiceScore) -> Mapping[str, float]:
    return dict(zip(score.labels, score.log_probabilities))
