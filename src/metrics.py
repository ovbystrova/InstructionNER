from typing import Dict, List, Tuple, Optional

import numpy as np


# TODO remove Tuple Spans with dataclass Span


def calculate_span_metrics_per_batch(
        spans_pred: List[List[Tuple[int, int, str]]],
        spans_true: List[List[Tuple[int, int, str]]],
        options: List[str]
):
    metrics_per_type = {}
    confusion_matrix_per_type = {option: get_empty_confusion_matrix() for option in options}

    for spans_pred_text, spans_true_text in zip(spans_pred, spans_true):

        confusion_matrix_text = calculate_confusion_matrix(
            spans_pred=spans_pred_text,
            spans_true=spans_true_text,
            confusion_matrix_per_type=confusion_matrix_per_type
        )

        for option in options:
            confusion_matrix_per_type[option] = {key: confusion_matrix_per_type.get(key, 0) +
                                                      confusion_matrix_text.get(key, 0)
                                                 for key in set(confusion_matrix_per_type) | set(confusion_matrix_text)
                                                 }

    for option in options:
        metrics_per_type[option] = calculate_metrics_from_confusion_matrix(
            confusion_matrix=confusion_matrix_per_type[option]
        )

    metrics = average_metrics(
        metrics=metrics_per_type,
        confusion_matrix=confusion_matrix_per_type
    )

    return metrics, metrics_per_type, confusion_matrix_per_type


def calculate_confusion_matrix(
        spans_pred: List[Tuple[int, int, str]],
        spans_true: List[Tuple[int, int, str]],
        confusion_matrix_per_type: Optional[Dict[str, Dict[str, int]]]
):
    if confusion_matrix_per_type is None:
        labels = set([span[-1] for span in spans_true]) & set([span[-1] for span in spans_pred])
        confusion_matrix_per_type = {label: get_empty_confusion_matrix() for label in labels}

    spans_true_missed_in_pred = set(spans_true) - set(spans_pred)

    for span_pred in spans_pred:

        start, end, label = span_pred

        spans_true_missed_in_pred_label = [span for span in spans_true_missed_in_pred if span[-1] == label]

        if span_pred in spans_true:
            confusion_matrix_per_type[label]["TP"] += 1

        else:
            equal_start = [span for span in spans_true if span[0] == start and span[1] != end and span[-1] == label]
            equal_end = [span for span in spans_true if span[1] == end and span[-1] == label and span[0] != start]

            for equal_start_span in equal_start:
                _end = equal_start_span[1]
                if end < _end:
                    confusion_matrix_per_type[label]["FN"] += 1
                elif end > _end:
                    confusion_matrix_per_type[label]["FP"] += 1

                if equal_start_span in spans_true_missed_in_pred_label:
                    spans_true_missed_in_pred_label.remove(equal_start_span)

            for equal_end_span in equal_end:
                _start = equal_end_span[1]
                if start > _start:
                    confusion_matrix_per_type[label]["FN"] += 1
                elif start < _start:
                    confusion_matrix_per_type[label]["FP"] += 1

                if equal_end_span in spans_true_missed_in_pred_label:
                    spans_true_missed_in_pred_label.remove(equal_end_span)

        confusion_matrix_per_type[label]["FN"] += len(spans_true_missed_in_pred_label)

    return confusion_matrix_per_type


def get_empty_confusion_matrix():
    matrix = {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "TN": 0
    }
    return matrix


def calculate_metrics_from_confusion_matrix(
        confusion_matrix: Dict[str, int]
):
    precision = confusion_matrix["TP"] / (confusion_matrix["TP"] + confusion_matrix["FP"])
    recall = confusion_matrix["TP"] / (confusion_matrix["TP"] + confusion_matrix["FN"])
    f1score = 2 * precision * recall / (precision + recall)

    metrics = {
        "f1-score": f1score,
        "precision": precision,
        "recall": recall
    }

    return metrics


def average_metrics(
        metrics: Dict[str, Dict[str, float]],
        confusion_matrix: Dict[str, Dict[str, float]]
):
    metrics_average = {}

    precisions = []
    recalls = []
    f1scores = []

    tps = []
    fps = []
    fns = []
    supports = []

    for label, metrics_per_label in metrics.items():
        precisions.append(metrics_per_label["precision"])
        recalls.append(metrics_per_label["recall"])
        f1scores.append(metrics_per_label["f1-score"])

        confusion_matrix_per_type = confusion_matrix[label]

        tp = confusion_matrix_per_type["TP"]
        tps.append(tp)

        fps.append(confusion_matrix_per_type["FP"])

        fn = confusion_matrix_per_type["FN"]
        fns.append(fn)

        supports.append(tp + fn)

    metrics_average["precision_macro"] = np.mean(precisions)
    metrics_average["recall_macro"] = np.mean(recalls)
    metrics_average["f1-score_macro"] = np.mean(f1scores)

    metrics_average["precision_micro"] = np.sum(tps) / (np.sum(tps) + np.sum(fps))
    metrics_average["recall_micro"] = np.sum(tps) / (np.sum(tps) + np.sum(fns))
    metrics_average["f1-score_micro"] = 2 * metrics_average["precision_micro"] * metrics_average["recall_micro"] \
                                        / (metrics_average["precision_micro"] + metrics_average["recall_micro"])

    metrics_average["precision_weighted"] = np.average(precisions, weights=supports)
    metrics_average["recall_weighted"] = np.average(recalls, weights=supports)
    metrics_average["f1-score_weighted"] = np.average(f1scores, weights=supports)

    return metrics_average
