import re
from typing import List, Optional

from instruction_ner.core.datatypes import Preffix, Span


class PredictionSpanFormatter:
    """
    Turns raw Model output into NER spans (start_idx, end_idx, label)
    """

    answer_templates = ["is an", "is a"]  # TODO move this (get rid of literals)

    def format_answer_spans(
        self, context: str, prediction: str, options: List[str]
    ) -> List[Span]:
        """
        Based on model prediction and context create entity spans
        :param options:
        :param context:
        :param prediction:
        :return:
        """

        entity_spans = []
        source_sentence = context.replace(
            Preffix.CONTEXT.value,
            "",
            1  # replace only the first occurrence of substring
        )

        prediction = prediction.strip(
            "."
        )  # Because answer in train data always ends with '.'
        prediction_parts = prediction.split(",")

        for prediction_part in prediction_parts:
            spans = self._get_span_from_part(prediction_part, source_sentence)
            if spans is None:
                continue

            spans = [span for span in spans if span.label in options]
            entity_spans.extend(spans)

        return entity_spans

    def _get_span_from_part(
        self, prediction_part: str, source_sentence: str
    ) -> Optional[List[Span]]:
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

            try:
                matches = list(re.finditer(value, source_sentence))
            except re.error:  # unbalanced parenthesis at position
                return None

            if len(matches) == 0:
                return None

            spans = []

            for match in matches:

                start = match.start()
                end = match.end()
                span = Span(start=start, end=end, label=label)
                spans.append(span)

            return spans

        return None
