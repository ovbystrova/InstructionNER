from typing import Dict, List

from transformers import T5Tokenizer, T5ForConditionalGeneration

from src.core.datatypes import TaskType
from src.formatters import (
    EntityTypeTaskFormatter,
    NERTaskFormatter,
    EntityExtractTaskFormatter,
    PredictionSpanFormatter
)


class Model:

    def __init__(self,
                 model_path: str,
                 tokenizer_path: str,
                 instructions: Dict[str, str],
                 options: List[str]
                 ):

        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)

        self.task_to_formatter = {
            TaskType.NER: NERTaskFormatter,
            TaskType.ENTITY_EXTRACTOR: EntityExtractTaskFormatter,
            TaskType.ENTITY_TYPING: EntityTypeTaskFormatter
        }
        self.answer_formatter = PredictionSpanFormatter()

        self.instructions = instructions
        self.options = options

    # TODO add PredictionList formatter for TaskType.EntitiesExtractor
    # TODO change return to Any task, not only spans
    def predict(self, text: str, task_type=TaskType.NER):
        """
        Generate prediction and format spans based on TaskType
        :param text: input text
        :param task_type: one of NER / EntityExtractor / EntityTyping
        :return:
        """

        if task_type not in self.task_to_formatter:
            raise ValueError(f"Expected task_type to be on of {self.task_to_formatter.keys()}")

        # TODO this is dirty. Think of more elegant inference
        data = {"context": text, "entities": []}

        instance = self.task_to_formatter[task_type].format_instance(
            data=data,
            instruction=self.instructions[task_type.value],
            options=self.options
        )

        input_ids = self.tokenizer(
            [instance.context], [instance.question], return_tensors="pt").input_ids

        outputs = self.model.generate(input_ids)
        # change to false if labels are special tokens
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        answer_spans = self.answer_formatter.format_answer_spans(
            instance=instance,
            prediction=answer
        )

        return answer_spans
