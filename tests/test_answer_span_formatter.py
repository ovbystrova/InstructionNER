import json
from pathlib import Path
from unittest import TestCase

from parameterized import parameterized

from src.core.datatypes import Instance
from src.formatters import PredictionSpanFormatter


def _make_instance(data):
    instance = Instance(
        context=data["context"],
        question=data["question"],
        answer=data["answer"]
    )
    return instance


class TestSpanFormatter(TestCase):

    maxDiff = None
    test_data_dir = Path(__file__).parent / "data"
    with open(test_data_dir / "test_case_answer_formatter.json") as f:
        data = json.load(f)

    instance = _make_instance(data["instance"])
    predictions = data["answers"]

    spans_true = []
    for spans in data["spans_true"]:
        spans = [tuple(span) for span in spans]
        spans_true.append(spans)

    @parameterized.expand([
        (instance, predictions[0], spans_true[0]),
        (instance, predictions[0], spans_true[0]),
        (instance, predictions[0], spans_true[0])
    ])
    def test_format_span(self, instance, prediction, spans_true):

        formatter = PredictionSpanFormatter()

        spans_pred = formatter.format_answer_spans(
            prediction=prediction,
            instance=instance
        )

        self.assertListEqual(
            spans_pred,
            spans_true
        )
