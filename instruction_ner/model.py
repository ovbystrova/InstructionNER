from typing import Any, Dict, List, Tuple

from dataclasses import astuple
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from instruction_ner.formatters import (
    NERTaskFormatter,
    PredictionSpanFormatter
)


class Model:

    def __init__(self,
                 model_path_or_name: str,
                 tokenizer_path_or_name: str
                 ):

        self.tokenizer = T5Tokenizer.from_pretrained(model_path_or_name,
                                                     # local_files_only=True
                                                     )
        self.model = T5ForConditionalGeneration.from_pretrained(tokenizer_path_or_name,
                                                                # local_files_only=True
                                                                )

        self.formatter = NERTaskFormatter()

        self.answer_formatter = PredictionSpanFormatter()

    def predict(self,
                text: str,
                generation_kwargs: Dict[str, Any],
                instruction: str,
                options: List[str]
                ) -> Tuple[str, List[Tuple[int, int, str]]]:
        """
        Generate prediction and format spans based on TaskType
        :param options:
        :param instruction:
        :param generation_kwargs:
        :param text: input text
        :return:
        """

        self.model.eval()

        instance = self.formatter.format_instance(
            context=text,
            instruction=instruction,
            options=options,
            entity_spans=None,
            entity_values=None
        )

        input_ids = self.tokenizer(
            [instance.context], [instance.question], return_tensors="pt").input_ids

        with torch.no_grad():
            outputs = self.model.generate(input_ids, **generation_kwargs)

        # change to false if labels are special tokens
        answer_raw = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        answer_spans = self.answer_formatter.format_answer_spans(
            context=instance.context,
            prediction=answer_raw,
            options=options
        )
        answer_spans = [astuple(span) for span in answer_spans]

        return answer_raw, answer_spans
