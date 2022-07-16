from typing import Dict, List, Tuple

import numpy as np


def calculate_metrics(
        spans_pred: List[List[Tuple[int, int, str]]],
        spans_true: List[List[Tuple[int, int, str]]],
        options: List[str]
):

    label2index = {"O": 0}
    for option in options:
        label2index[option] = len(label2index)

    confusion_matrix = build_confusion_matrix(
        spans_pred=spans_pred,
        spans_true=spans_true,
        label2index=label2index
    )

    metrics_per_label = calculate_metrics_from_confusion_matrix(
        confusion_matrix=confusion_matrix,
        label2index=label2index
    )

    metrics = add_average_metrics(
        confusion_matrix=confusion_matrix,
        label2index=label2index,
        metrics_per_label=metrics_per_label
    )

    return metrics


def build_confusion_matrix(
        spans_pred: List[List[Tuple[int, int, str]]],
        spans_true: List[List[Tuple[int, int, str]]],
        label2index: Dict[str]
) -> np.array:

    confusion_matrix = np.zeros((len(label2index), len(label2index)))

    for spans_pred_batch, spans_true_batch in zip(spans_pred, spans_true):

        confusion_matrix = update_confusion_matrix(
            spans_pred=spans_pred_batch,
            spans_true=spans_true_batch,
            confusion_matrix=confusion_matrix,
            label2index=label2index
        )

    return confusion_matrix


def update_confusion_matrix(
        spans_pred: List[Tuple[int, int, str]],
        spans_true: List[Tuple[int, int, str]],
        confusion_matrix: np.array,
        label2index: Dict[str]
) -> np.array:

    # i  - true
    # j - pred

    spans_true_missed_in_pred = set(spans_true) - set(spans_pred)

    for span_pred in spans_pred:

        start, end, label = span_pred
        j = label2index[label]

        if span_pred in spans_true:
            confusion_matrix[j][j] += 1  # True Positive
            continue

        equal_start = [span for span in spans_true if span[0] == start and span[1] != end and span[-1] == label]
        equal_end = [span for span in spans_true if span[1] == end and span[-1] == label and span[0] != start]

        if len(equal_start) == 0 and len(equal_end) == 0:
            confusion_matrix[label2index["O"]][j] += 1  # False Positive   # TODO remove 'O' with special variable
            continue

        for equal_start_span in equal_start:
            end_true = equal_start_span[1]
            if end < end_true:
                confusion_matrix[j][label2index["O"]] += 1  # False Negative
            elif end > end_true:
                confusion_matrix[label2index["O"]][j] += 1  # False Positive

            if equal_start_span in spans_true_missed_in_pred:
                spans_true_missed_in_pred.remove(equal_start_span)

        for equal_end_span in equal_end:
            start_true = equal_end_span[1]
            if start > start_true:
                confusion_matrix[j][label2index["O"]] += 1  # False Negative
            elif start < start_true:
                confusion_matrix[label2index["O"]][j] += 1  # False Positive

            if equal_end_span in spans_true_missed_in_pred:
                spans_true_missed_in_pred.remove(equal_end_span)

        # Treat not predicted spans as False Negative
        confusion_matrix[j][label2index["O"]] += len(spans_true_missed_in_pred)

    return confusion_matrix


def calculate_metrics_from_confusion_matrix(
        confusion_matrix: np.array,
        label2index: Dict[str]
) -> Dict[str, Dict[str, float]]:

    metrics = {}

    for label, idx in label2index.items():

        if label == "O":
            continue

        metrics_per_label = {}

        true_positive = confusion_matrix[idx][idx]
        precision = true_positive / (np.sum(confusion_matrix[:][idx]))  # TODO check this
        recall = true_positive / (np.sum(confusion_matrix[idx][:]))

        metrics_per_label["precision"] = precision
        metrics_per_label["recall"] = recall
        metrics_per_label["f1-score"] = 2 * precision * recall / (precision + recall)
        metrics_per_label["support"] = np.sum(confusion_matrix[idx][:])

    return metrics


def add_average_metrics(
        confusion_matrix: np.array,
        label2index: Dict[str, int],
        metrics_per_label: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:

    metrics = {}

    precisions, recalls, f1scores, supports = [], [], [], []
    for label, metrics in metrics_per_label.items():
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        f1scores.append(metrics["f1-score"])
        supports.append(metrics["support"])
    supports_proportions = [support / np.sum(supports) for support in supports]

    metrics["macro_avg"] = {
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1-score": np.mean(f1scores)
    }

    idxs = [value for key, value in label2index.items() if key != "O"]
    true_positive_total = np.sum(np.diag(confusion_matrix))  # TODO simplify this
    false_positive_total = np.sum([np.sum(confusion_matrix[:][idx]) - confusion_matrix[idx][idx] for idx in idxs])
    false_negative_total = np.sum([np.sum(confusion_matrix[idx][:]) - confusion_matrix[idx][idx] for idx in idxs])
    precision_micro = true_positive_total / (true_positive_total + false_positive_total)
    recall_micro = true_positive_total / (true_positive_total + false_negative_total)

    metrics["micro_avg"] = {
        "precision": precision_micro,
        "recall": recall_micro,
        "f1-score": 2 * precision_micro * recall_micro / (precision_micro + recall_micro)
    }

    metrics["weighted_avg"] = {
        "precision": np.average(precisions, weights=supports_proportions),
        "recall": np.average(recalls, weights=supports_proportions),
        "f1-score": np.average(f1scores, weights=supports_proportions)
    }

    return metrics
