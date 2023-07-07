import numpy as np
import re

from nlaugmenter.interfaces.SentenceOperation import SentenceOperation
from nlaugmenter.tasks.TaskTypes import TaskType


class RandomSmartDeletion(SentenceOperation):
    tasks = [TaskType.TEXT_CLASSIFICATION, TaskType.TEXT_TO_TEXT_GENERATION]
    languages = ["en"]
    _default_forbidden_to_delete_words = [
        "i",
        "is",
        "there",
        "that",
        "you",
        "your",
        "he",
        "she",
        "has",
        "have",
    ]
    _default_forbidden_to_delete_regex = [r"\b\w*[0-9]+\w*\b", "[?!.,']"]

    def __init__(
        self,
        prob: int = 0.25,
        seed: int = 0,
        forbidden_to_delete_words: list[str] | None = None,
        forbidden_to_delete_regex: list[str] | None = None,
    ):
        super().__init__(seed)
        forbidden_to_delete_words = (
            self._default_forbidden_to_delete_words
            if forbidden_to_delete_words is None
            else forbidden_to_delete_words
        )
        forbidden_to_delete_regex = (
            self._default_forbidden_to_delete_regex
            if forbidden_to_delete_regex is None
            else forbidden_to_delete_regex
        )
        self.prob = prob
        self.forbidden_to_delete_words = forbidden_to_delete_words
        self.forbidden_to_delete_regex = [
            re.compile(rgx) for rgx in forbidden_to_delete_regex
        ]

    def random_deletion(self, text: str, seed: int = 0):
        np.random.seed(seed)
        text = np.array(text.split())
        N = len(text)
        mask = np.random.binomial(1, 1 - self.prob, N) == 1

        for idx, mask_value in enumerate(mask):
            # if the mask is false, check whether the word is allowed
            if not mask_value:
                word = text[idx]
                if word.lower() in self.forbidden_to_delete_words or any(
                    re.match(rgx, word)
                    for rgx in self.forbidden_to_delete_regex
                ):
                    mask[idx] = True

        text_tf = text[mask]
        text_tf = " ".join(text_tf)
        text_tf = (
            text_tf if len(text_tf) > 0 else text[np.random.randint(0, N - 1)]
        )
        return [text_tf]

    def generate(self, sentence: str):
        perturbed_texts = self.random_deletion(text=sentence, seed=self.seed)
        return perturbed_texts
