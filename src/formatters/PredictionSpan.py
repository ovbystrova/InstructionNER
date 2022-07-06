from typing import Dict, Any, List, Tuple, Optional

from src.core.datatypes import Instance, Preffix
from src.core.formatter import Formatter


class PredictionSpanFormatter(Formatter):
    """
    Turns raw Model output into NER spans (start_idx, end_idx, label)
    """
    answer_templates = ["is a", "is an"]

    def format_instance(
            self,
            data: Dict[str, Any],
            instruction,
            options
    ) -> Instance:
        raise NotImplementedError

    def format_answer_spans(self, instance: Instance, prediction: str) -> List[Tuple[int, int, str]]:
        """
        Based on model prediction and instance created entity spans
        :param instance:
        :param prediction:
        :return:
        """

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

    def _get_span_from_part(self, prediction_part: str, source_sentence: str) -> Optional[Tuple(int, int, str)]:
        """
        Gets entity span from part of prediction
        :param prediction_part: Olga is a PER
        :param source_sentence: Today Olga decided to sleep a lot.
        :return: (6, 10, "PER")
        """

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
