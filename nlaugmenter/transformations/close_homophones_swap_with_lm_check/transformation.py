import random
import spacy
from SoundsLike.SoundsLike import Search
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from nlaugmenter.interfaces.SentenceOperation import SentenceOperation
from nlaugmenter.tasks.TaskTypes import TaskType
from nlaugmenter.utils.initialize import spacy_nlp


def find_possible_homophone_replacements(
    token: str, strict: bool = False
) -> str:
    """Searches the possible homophone space for `token`. If `strict` is set, only perfect
    homophones will be searched, otherwise close homophones will be included as well.
    """
    exact_homophones = [h.lower() for h in Search.perfectHomophones(token)]
    if token.lower() in exact_homophones:
        exact_homophones.remove(token.lower())

    if strict:
        return exact_homophones

    close_homophones = [h.lower() for h in Search.closeHomophones(token)]
    if token.lower() in close_homophones:
        close_homophones.remove(token.lower())

    found_homophones = list(
        (set(exact_homophones)).union(set(close_homophones))
    )
    return found_homophones


class CloseHomophonesSwapWithLmCheck(SentenceOperation):
    tasks = [
        TaskType.TEXT_CLASSIFICATION,
        TaskType.TEXT_TO_TEXT_GENERATION,
        TaskType.TEXT_TAGGING,
    ]
    languages = ["en"]
    keywords = [
        "lexical",
        "rule-based",
        "high-coverage",
        "high-precision",
        "unnaturally-written",
    ]

    def __init__(self, seed: int = 0, max_outputs: int = 1, prob: float = 0.5):
        super().__init__(seed)
        self.max_outputs = max_outputs
        self.prob = prob
        self.forbidden_to_switch = ["a", "an", "I"]
        self.nlp = spacy_nlp if spacy_nlp else spacy.load("en_core_web_sm")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, sentence: str):
        perturbed_texts = self.close_homophones_swap(
            text=sentence,
            seed=self.seed,
            max_outputs=self.max_outputs,
            nlp=self.nlp,
        )
        return perturbed_texts

    def score_sentence(self, sentence: str) -> float:
        tokenize_input = self.tokenizer.tokenize(sentence)
        tensor_input = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(tokenize_input)]
        )
        loss = self.model(tensor_input, labels=tensor_input)[0]
        return loss.item()

    def choose_most_logical_homophone(
        self, index: int, homophone_list: list[str], sentence: list[str]
    ) -> int:
        sentence_scores = []
        for homophone_replacement in homophone_list:
            replaced_homophone_sentence = sentence[:]
            replaced_homophone_sentence[index] = homophone_replacement
            score = self.score_sentence(" ".join(replaced_homophone_sentence))
            sentence_scores.append(score)
        most_probable_value_idx = torch.argmin(
            torch.tensor(sentence_scores)
        ).item()
        return homophone_list[most_probable_value_idx]

    def close_homophones_swap(
        self,
        text: str,
        seed: int = 0,
        max_outputs: int = 1,
        nlp=None,
        strict_homophones: bool = False,
    ) -> str:
        random.seed(seed)
        doc = nlp(text)
        perturbed_texts = []
        spaces = [True if tok.whitespace_ else False for tok in doc]
        for _ in range(max_outputs):
            perturbed_text = []
            for index, token in enumerate(doc):
                if random.uniform(0, 1) < self.prob:
                    # check whether the token is not forbidden to be replaced
                    if token.text.lower() in self.forbidden_to_switch:
                        perturbed_text.append(token.text)
                        continue

                    try:
                        possible_replacements = (
                            find_possible_homophone_replacements(token.text)
                        )

                        # if there is only one homophone candidate, do not bother with LM scoring
                        if len(possible_replacements) > 1:
                            # create the homophones test sentence (include what was already perturbed)
                            test_sentence = (
                                perturbed_text + [t.text for t in doc][index:]
                            )
                            replacement = self.choose_most_logical_homophone(
                                index, possible_replacements, test_sentence
                            )
                        elif len(possible_replacements) == 1:
                            replacement = possible_replacements[0]
                        else:  # no good replacement was found, continue
                            perturbed_text.append(token.text)
                            continue

                        perturbed_text.append(replacement.lower())

                    except Exception as e:
                        perturbed_text.append(token.text)
                else:
                    perturbed_text.append(token.text)

            textbf = []
            for index, token in enumerate(perturbed_text):
                textbf.append(token)
                if spaces[index]:
                    textbf.append(" ")
            perturbed_texts.append("".join(textbf))
        return perturbed_texts
