from typing import Dict, List

import numpy as np

from instruction_ner.core.datatypes import Span


def calculate_metrics(
        spans_pred: List[List[Span]],
        spans_true: List[List[Span]],
        options: List[str]
):
    """
    Built confusion matrix and calculate metrics (precision, recall, f1-score)
    average options: micro, macro, weighted
    :param spans_pred: predicted spans over batch / epoch
    :param spans_true: true spans over batch / epoch
    :param options: list of labels presented in data
    :return: dictionary of per label metrics and average metrics
    """

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
        metrics=metrics_per_label
    )

    return metrics


def build_confusion_matrix(
        spans_pred: List[List[Span]],
        spans_true: List[List[Span]],
        label2index: Dict[str, int]
) -> np.array:
    """
    Build confusion matrix based on true and predicted spans
    :param spans_pred: predicted spans over batch / epoch
    :param spans_true: true spans over batch / epoch
    :param label2index: mapping between label and its index in confusion matrix
    :return: confusion matrix
    """

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
        spans_pred: List[Span],
        spans_true: List[Span],
        confusion_matrix: np.array,
        label2index: Dict[str, int]
) -> np.array:
    """
    Update confusion matrix based on spans from one data item
    :param spans_pred: predicted spans from text
    :param spans_true: true spans for text
    :param confusion_matrix: matrix (n_labels, n_labels)
    :param label2index: mapping between label and its index in confusion matrix
    :return: confusion matrix
    """

    # i  - true
    # j - pred

    spans_true_missed_in_pred = set(spans_true) - set(spans_pred)

    for span_pred in spans_pred:

        j = label2index[span_pred.label]

        if span_pred in spans_true:
            confusion_matrix[j][j] += 1  # True Positive
            continue

        equal_start = [span for span in spans_true if span.start == span_pred.start and span.end != span_pred.end
                       and span.label == span_pred.label]
        equal_end = [span for span in spans_true if span.end == span_pred.end and span.label == span_pred.label
                     and span.start != span_pred.start]
        equal_start_end = [span for span in spans_true if span.end == span_pred.end and span.start == span_pred.start
                           and span.label != span_pred.label]

        if len(equal_start_end) > 0:  # If model found the right boundaries but wrong label
            equal_start_end_span = equal_start_end[0]
            confusion_matrix[label2index[equal_start_end_span.label]][j] += 1
            if equal_start_end_span in spans_true_missed_in_pred:
                spans_true_missed_in_pred.remove(equal_start_end_span)
            continue

        elif len(equal_start) == 0 and len(equal_end) == 0:  # If model found wrong boundaries
            confusion_matrix[label2index["O"]][j] += 1  # False Positive   # TODO remove 'O' with special variable
            continue

        for equal_start_span in equal_start:  # If start matches  # TODO simplify
            if span_pred.end < equal_start_span.end:
                confusion_matrix[j][label2index["O"]] += 1  # False Negative
            elif span_pred.end > equal_start_span.end:
                confusion_matrix[label2index["O"]][j] += 1  # False Positive

            if equal_start_span in spans_true_missed_in_pred:
                spans_true_missed_in_pred.remove(equal_start_span)

        for equal_end_span in equal_end:  # If end matches
            if span_pred.start > equal_end_span.start:
                confusion_matrix[j][label2index["O"]] += 1  # False Negative
            elif span_pred.start < equal_end_span.start:
                confusion_matrix[label2index["O"]][j] += 1  # False Positive

            if equal_end_span in spans_true_missed_in_pred:
                spans_true_missed_in_pred.remove(equal_end_span)

    # Treat not predicted spans as False Negative
    for span in spans_true_missed_in_pred:
        confusion_matrix[label2index[span.label]][label2index["O"]] += 1

    return confusion_matrix


def calculate_metrics_from_confusion_matrix(
        confusion_matrix: np.array,
        label2index: Dict[str, int]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate Precision, Recall, F1-Score per label (without average)
    :param confusion_matrix: np.array
    :param label2index: mapping between label and its index in confusion matrix
    :return:
    """

    metrics = {}

    for label, idx in label2index.items():

        if label == "O":
            continue

        metrics_per_label = {}

        true_positive = confusion_matrix[idx][idx]
        precision = true_positive / (np.sum(confusion_matrix[:, idx])) if true_positive > 0 else 0
        recall = true_positive / (np.sum(confusion_matrix[idx, :])) if true_positive > 0 else 0

        metrics_per_label["precision"] = precision
        metrics_per_label["recall"] = recall
        metrics_per_label["f1-score"] = 2 * precision * recall / (precision + recall) \
            if precision > 0 and recall > 0 else 0
        metrics_per_label["support"] = np.sum(confusion_matrix[idx][:])

        metrics[label] = metrics_per_label

    return metrics


def add_average_metrics(
        confusion_matrix: np.array,
        label2index: Dict[str, int],
        metrics: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Add micro average, marco average, and weighted average metrics
    :param confusion_matrix: np.array
    :param label2index: mapping between label and its index in confusion matrix
    :param metrics: metrics per label
    :return:
    """

    precisions, recalls, f1scores, supports = [], [], [], []

    for label, metrics_label in metrics.items():

        precisions.append(metrics_label["precision"])
        recalls.append(metrics_label["recall"])
        f1scores.append(metrics_label["f1-score"])
        supports.append(metrics_label["support"])

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
        if precision_micro > 0 and recall_micro > 0 else 0
    }

    metrics["weighted_avg"] = {
        "precision": np.average(precisions, weights=supports_proportions),
        "recall": np.average(recalls, weights=supports_proportions),
        "f1-score": np.average(f1scores, weights=supports_proportions)
    }

    return metrics
