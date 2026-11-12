import unittest

import torch

from subtext.forced_choice import (
    ForcedChoiceError,
    score_single_token_answers,
    score_answer_sequences,
    semantic_log_odds,
    single_token_answers,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"A": [3], "B": [7], "AB": [3, 7], "left": [3], "right": [4]}[text]


class FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    def __call__(self, *, input_ids, use_cache):
        del use_cache
        logits = torch.zeros((1, input_ids.shape[1], 8))
        logits[:, :, 3] = 2.0
        return FakeOutput(logits)


class ForcedChoiceTests(unittest.TestCase):
    def test_scores_and_semantically_aligns_answer_logits(self):
        logits = torch.zeros(10)
        logits[3] = 2.0
        logits[7] = 0.5
        score = score_single_token_answers(
            torch, logits, labels=("A", "B"), token_ids=(3, 7)
        )
        self.assertEqual(score.chosen_label, "A")
        self.assertAlmostEqual(score.raw_log_odds, 1.5, places=6)
        self.assertAlmostEqual(
            semantic_log_odds(score, positive_label="B"), -1.5, places=6
        )

    def test_answer_tokenization_fails_closed(self):
        self.assertEqual(single_token_answers(FakeTokenizer(), ("A", "B")), (3, 7))
        with self.assertRaises(ForcedChoiceError):
            single_token_answers(FakeTokenizer(), ("A", "AB"))

    def test_scores_answer_sequences_without_sampling(self):
        score = score_answer_sequences(
            torch,
            FakeModel(),
            torch.tensor([[1, 2]]),
            tokenizer=FakeTokenizer(),
            labels=("left", "right"),
        )
        self.assertEqual(score.chosen_label, "left")
        self.assertGreater(score.raw_log_odds, 0.0)


if __name__ == "__main__":
    unittest.main()
