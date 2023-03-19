import json
from dataclasses import astuple
from pathlib import Path
from unittest import TestCase

from parameterized import parameterized

from instruction_ner.formatters import PredictionSpanFormatter


class TestSpanFormatter(TestCase):

    maxDiff = None
    test_data_dir = Path(__file__).parent / "data"
    with open(test_data_dir / "test_case_prediction_span_formatter.json") as f:
        data = json.load(f)

    context = data["context"]
    predictions = data["answers"]

    spans_true = []
    for spans in data["spans_true"]:
        spans = [tuple(span) for span in spans]
        spans_true.append(spans)

    @parameterized.expand(
        [
            (context, predictions[0], spans_true[0]),
            (context, predictions[1], spans_true[1]),
            (context, predictions[2], spans_true[2])
        ]
    )
    def test_format_span(self, context, prediction, spans_true):

        formatter = PredictionSpanFormatter()
        options = ["ORG", "PER", "LOC"]

        spans_pred = formatter.format_answer_spans(
            context=context, prediction=prediction, options=options
        )

        spans_pred = [astuple(span) for span in spans_pred]

        self.assertListEqual(spans_pred, spans_true)
