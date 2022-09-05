import json
import random
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from instruction_ner.formatters.PredictionSpan import PredictionSpanFormatter
from instruction_ner.metrics import calculate_metrics

prediction_span_formatter = PredictionSpanFormatter()


def evaluate(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    dataloader: torch.utils.data.DataLoader,
    writer: Optional[SummaryWriter],
    device: torch.device,
    generation_kwargs: Dict[str, Any],
    epoch: int,
    options: List[str],
):
    model.eval()

    epoch_loss = []

    spans_true = []
    spans_pred = []

    with torch.no_grad():
        for i, inputs in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="Evaluating",
        ):

            instances = inputs.pop("instances")
            contexts = [instance.context for instance in instances]
            spans_true_batch = [instance.entity_spans for instance in instances]
            spans_true.extend(spans_true_batch)

            answers = inputs.pop("answers")

            # replace padding token id's of the labels by -100 so it's ignored by the loss
            answers = torch.tensor(answers.input_ids)
            answers[answers == tokenizer.pad_token_id] = -100

            inputs.to(device)
            answers = answers.to(device)
            outputs = model(**inputs, labels=answers)
            loss = outputs.loss

            prediction_texts = model.generate(**inputs, **generation_kwargs)
            prediction_texts = tokenizer.batch_decode(
                prediction_texts, skip_special_tokens=True
            )
            if writer:
                writer.add_text("sample_prediction", prediction_texts[0])

            spans_pred_batch = [
                prediction_span_formatter.format_answer_spans(
                    context, prediction, options
                )
                for context, prediction in zip(contexts, prediction_texts)
            ]
            spans_pred.extend(spans_pred_batch)

            batch_metrics = calculate_metrics(
                spans_pred_batch, spans_true_batch, options=options
            )

            if writer:
                for metric_class, metric_dict in batch_metrics.items():
                    writer.add_scalars(
                        metric_class, metric_dict, epoch * len(dataloader) + i
                    )

            epoch_loss.append(loss.item())

            if writer:
                writer.add_scalar(
                    "batch loss / evaluation", loss.item(), epoch * len(dataloader) + i
                )

        epoch_metrics = calculate_metrics(spans_pred, spans_true, options=options)

        show_classification_report(epoch_metrics)

        return epoch_metrics


def get_sample_text_prediction(
    model: T5ForConditionalGeneration,
    dataloader: torch.utils.data.DataLoader,
    tokenizer: T5Tokenizer,
    generation_kwargs,
    device: str,
    options: List[str],
    n: int = 3,
):
    """
    Generate sample N predictions
    :param model:
    :param dataloader:
    :param tokenizer:
    :param generation_kwargs: arguments for generation process
    :param device: cuda or cpu
    :param options: list of labels
    :param n: number of prediction to generate
    :return:
    """
    model.eval()

    dataset = dataloader.dataset

    ids_to_pick = random.sample(list(range(0, len(dataset))), n)

    for _id in ids_to_pick:
        dataset_item = dataset[_id]

        print(f"Input: {dataset_item.context}")
        print(f"{dataset_item.question}")

        input_ids = tokenizer(
            [dataset_item.context], [dataset_item.question], return_tensors="pt"
        ).input_ids

        input_ids = input_ids.to(device)
        answer = model.generate(input_ids, **generation_kwargs)
        answer = tokenizer.decode(
            answer[0], skip_special_tokens=True
        )  # TODO change to false

        answer_spans = prediction_span_formatter.format_answer_spans(
            context=dataset_item.context, prediction=answer, options=options
        )

        print(f"Prediction: {answer}")
        print(f"Found {len(answer_spans)} spans. {answer_spans}\n")


def update_best_checkpoint(
    metrics_new: Dict[str, Dict[str, float]],
    metrics_best: Dict[str, Dict[str, float]],
    metric_name: str,
    metric_avg: str,
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    path_to_save_model: Optional[str],
):
    """
    Compares specific metric in two metric dictionaries: current and best.
    If new metric value is better -> new best model checkpoint saved
    :param metrics_new:
    :param metrics_best:
    :param metric_name:
    :param metric_avg:
    :param model:
    :param tokenizer:
    :param path_to_save_model:
    :return:
    """

    metric_current_value = metrics_new[metric_avg][metric_name]

    metric_best_value = 0.0
    if len(metrics_best) > 0:
        metric_best_value = metrics_best[metric_avg][metric_name]

    if metric_current_value > metric_best_value:
        print(
            f"Got Better results for {metric_name}. \n"
            f"{metric_current_value} > {metric_best_value}. Updating the best checkpoint"
        )
        metrics_best = metrics_new

        model.save_pretrained(path_to_save_model)
        tokenizer.save_pretrained(path_to_save_model)

        if path_to_save_model is not None:
            save_metrics_path = path_to_save_model + "/metrics.json"
        else:
            save_metrics_path = "metrics.json"

        with open(save_metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_best, ensure_ascii=False, indent=4, fp=f)

    return metrics_best


def show_classification_report(metrics: Dict[str, Dict[str, float]]):
    """
    Based on dictionary of metrics show classification report aka sklearn
    :param metrics:
    :return:
    """
    df = pd.DataFrame.from_dict(metrics)
    print(df.transpose())
