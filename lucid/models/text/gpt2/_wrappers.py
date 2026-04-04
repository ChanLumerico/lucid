from lucid.models.text.gpt import (
    GPTLMHeadModel,
    GPTDoubleHeadsModel,
    GPTForSequenceClassification,
)

from ._model import GPT2, GPT2Config

__all__ = ["GPT2LMHeadModel", "GPT2DoubleHeadsModel", "GPT2ForSequenceClassification"]


class GPT2LMHeadModel(GPTLMHeadModel):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__(config)
        self.gpt = GPT2(config)
        self.tie_weights()


class GPT2DoubleHeadsModel(GPTDoubleHeadsModel):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__(config)
        self.gpt = GPT2(config)
        self.tie_weights()


class GPT2ForSequenceClassification(GPTForSequenceClassification):
    def __init__(self, config: GPT2Config, num_labels: int = 2) -> None:
        super().__init__(config, num_labels)
        self.gpt = GPT2(config)
