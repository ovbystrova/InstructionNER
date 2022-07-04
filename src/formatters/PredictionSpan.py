from typing import Dict, Any, List, Tuple

from src.core.datatypes import Instance, Preffix
from src.core.formatter import Formatter


class PredictionSpanFormatter(Formatter):
    answer_templates = ["is a", "is an"]

    def format_instance(
            self,
            data: Dict[str, Any],
            instruction,
            options
    ) -> Instance:
        raise NotImplementedError

    def format_answer_spans(self, instance: Instance, prediction: str) -> List[Tuple[int, int, str]]:

        entity_spans = []
        source_sentence = instance.context.lstrip(Preffix.CONTEXT.value)

        prediction = prediction.strip(".")  # Because answer in train data always ends with '.'
        prediction_parts = prediction.split(",")

        for prediction_part in prediction_parts:
            span = self._get_span_from_part(
                prediction_part,
                source_sentence
            )
            if span is None:
                continue
            entity_spans.append(span)

        return entity_spans

    def _get_span_from_part(self, prediction_part, source_sentence):

        if not any([template in prediction_part for template in self.answer_templates]):
            return None

        for answer_template in self.answer_templates:
            _prediction_part = prediction_part.split(answer_template, maxsplit=2)

            if len(_prediction_part) != 2:
                continue

            value, label = _prediction_part[0], _prediction_part[1]
            value = value.strip(" ").rstrip(" ")
            label = label.strip(" ").rstrip(" ")

            print(source_sentence)
            value_counts_in_sentence = source_sentence.count(value)

            if value_counts_in_sentence == 0:
                return None

            elif value_counts_in_sentence > 1:
                raise ValueError(f"Expected to be one value per sentence, found {value_counts_in_sentence}")

            start = source_sentence.find(value)
            end = start + len(value)
            span = (start, end, label)
            return span

        return None
