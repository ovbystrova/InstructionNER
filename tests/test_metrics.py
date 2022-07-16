import json
from pathlib import Path
from unittest import TestCase

from parameterized import parameterized

from src.metrics import calculate_metrics


class TestMetrics(TestCase):
    maxDiff = None
    test_data_dir = Path(__file__).parent / "data" / "metrics"

    @parameterized.expand([
        (
                [[(0, 5, "ORG"), (14, 19, "PER")], [(34, 44, "LOC")]],
                [[(0, 5, "ORG"), (14, 19, "PER")], [(34, 44, "LOC")]],
                test_data_dir / "metrics_everything_correct.json",
                ["ORG", "LOC", "PER"]
        )
    ])
    def test_everything_correct(self, spans_pred, spans_true, metrics_true_filepath, options):

        metrics_pred = calculate_metrics(
            spans_pred=spans_pred,
            spans_true=spans_true,
            options=options
        )

        metrics_true = self._load_metrics_from_json(
            filepath=metrics_true_filepath
        )

        self.assertDictEqual(
            metrics_pred,
            metrics_true
        )

    @parameterized.expand([
        (
                [[(0, 5, "LOC"), (1, 10, "PER")], [(34, 44, "PER")]],
                [[(0, 5, "ORG"), (14, 19, "PER")], [(56, 59, "LOC")]],
                test_data_dir / "metrics_everything_wrong.json",
                ["ORG", "LOC", "PER"]
        )
    ])
    def test_everything_wrong(self, spans_true, spans_pred, metrics_true_filepath, options):

        metrics_pred = calculate_metrics(
            spans_pred=spans_pred,
            spans_true=spans_true,
            options=options
        )

        metrics_true = self._load_metrics_from_json(
            filepath=metrics_true_filepath
        )

        self.assertDictEqual(
            metrics_pred,
            metrics_true
        )

    @parameterized.expand([
        (
                [[(0, 5, "LOC"), (1, 10, "PER")], [(34, 44, "PER")], [(34, 44, "PER"), (1, 10, "PER")]],
                [[(0, 3, "LOC"), (3, 10, "PER")], [(56, 59, "ORG"), (34, 44, "PER")], [(32, 44, "PER"), (1, 14, "PER")]],
                test_data_dir / "metrics_equals_start_end.json",
                ["ORG", "LOC", "PER"]
        )
    ])
    def test_equals_start_end(self, spans_true, spans_pred, metrics_true_filepath, options):

        metrics_pred = calculate_metrics(
            spans_pred=spans_pred,
            spans_true=spans_true,
            options=options
        )

        metrics_true = self._load_metrics_from_json(
            filepath=metrics_true_filepath
        )

        self.assertDictEqual(
            metrics_pred,
            metrics_true
        )

    @staticmethod
    def _load_metrics_from_json(filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)
        return data

