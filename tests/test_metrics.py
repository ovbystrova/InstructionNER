import json
from pathlib import Path
from unittest import TestCase

from parameterized import parameterized

from src.metrics import calculate_metrics


class TestMetrics(TestCase):
    maxDiff = None
    test_data_dir = Path(__file__).parent / "data" / "metrics"
    with open(test_data_dir / "test_cases_metrics.json") as f:
        data_spans = json.load(f)

    metrics_true_files = [file for file in test_data_dir.rglob("*.json") if file.name.startswith("metrics_true_")]

    spans_true = []
    for spans in data_spans["spans_true"]:
        spans = [tuple(span) for span in spans]
        spans_true.append(spans)

    spans_pred = []
    for spans in data_spans["spans_pred"]:
        spans = [tuple(span) for span in spans]
        spans_pred.append(spans)

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

        print(metrics_pred)

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

        print(metrics_pred)

        metrics_true = self._load_metrics_from_json(
            filepath=metrics_true_filepath
        )

        with open(metrics_true_filepath, "w", encoding="utf-8") as f:
            json.dump(metrics_pred, indent=4, fp=f, ensure_ascii=False)

        self.assertDictEqual(
            metrics_pred,
            metrics_true
        )


    @staticmethod
    def _load_metrics_from_json(filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)
        return data

