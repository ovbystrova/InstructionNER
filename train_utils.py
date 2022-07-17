import random
from typing import List, Dict, Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.metrics import calculate_metrics
from src.formatters.PredictionSpan import PredictionSpanFormatter
from src.utils import show_classification_report

prediction_span_formatter = PredictionSpanFormatter()


def train(
        n_epochs: int,
        model: T5ForConditionalGeneration,
        tokenizer: T5Tokenizer,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        writer: SummaryWriter,
        device: torch.device,
        eval_every_n_batches: int,
        pred_every_n_batches: int,
        generation_kwargs: Dict[str, Any],
        options: List[str]
) -> None:
    for epoch in range(n_epochs):
        print(f"Epoch [{epoch + 1} / {n_epochs}]\n")

        train_epoch(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            writer=writer,
            device=device,
            epoch=epoch,
            eval_every_n_batches=eval_every_n_batches,
            pred_every_n_batches=pred_every_n_batches,
            generation_kwargs=generation_kwargs,
            options=options
        )

        evaluate(
            model=model,
            tokenizer=tokenizer,
            dataloader=test_dataloader,
            writer=writer,
            device=device,
            epoch=epoch,
            generation_kwargs=generation_kwargs,
            options=options
        )


def train_epoch(
        model: T5ForConditionalGeneration,
        tokenizer: T5Tokenizer,
        train_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        writer: SummaryWriter,
        device: torch.device,
        epoch: int,
        eval_every_n_batches,
        pred_every_n_batches,
        generation_kwargs,
        options: List[str],
        test_dataloader: torch.utils.data.DataLoader = None,

) -> None:
    """
    One training cycle (loop).
    Args:
        :param options: list of labels in dataset
        :param generation_kwargs:
        :param test_dataloader:
        :param train_dataloader:
        :param pred_every_n_batches: do sample prediction every n batches
        :param eval_every_n_batches: do evaluation every n batches
        :param model:
        :param optimizer:
        :param writer:
        :param epoch: current epoch
        :param device:
        :param tokenizer:
    """

    epoch_loss = []

    for i, inputs in tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc="Training",
    ):
        model.train()
        optimizer.zero_grad()

        inputs.pop("instances")
        answers = inputs.pop("answers")

        # replace padding token id's of the labels by -100 so it's ignored by the loss
        answers = torch.tensor(answers.input_ids)
        answers[answers == tokenizer.pad_token_id] = -100

        inputs.to(device)
        answers = answers.to(device)
        outputs = model(**inputs, labels=answers)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        writer.add_scalar(
            "batch loss / train", loss.item(), epoch * len(train_dataloader) + i
        )

        if i % eval_every_n_batches == 0 and i >= eval_every_n_batches:
            if test_dataloader is not None:
                evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    dataloader=test_dataloader,
                    writer=writer,
                    device=device,
                    epoch=epoch,
                    generation_kwargs=generation_kwargs,
                    options=options
                )
        if i % pred_every_n_batches == 0 and i >= pred_every_n_batches:
            get_sample_text_prediction(
                model=model,
                tokenizer=tokenizer,
                dataloader=test_dataloader,
                device=device,
                generation_kwargs=generation_kwargs
            )

    avg_loss = np.mean(epoch_loss)
    print(f"Train loss: {avg_loss}\n")
    writer.add_scalar("loss / train", avg_loss, epoch)


def evaluate(
        model: T5ForConditionalGeneration,
        tokenizer: T5Tokenizer,
        dataloader: torch.utils.data.DataLoader,
        writer: SummaryWriter,
        device: torch.device,
        generation_kwargs: Dict[str, Any],
        epoch: int,
        options: List[str]
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
            prediction_texts = tokenizer.batch_decode(prediction_texts, skip_special_tokens=True)
            writer.add_text("sample_prediction", prediction_texts[0])

            spans_pred_batch = [prediction_span_formatter.format_answer_spans(context, prediction, options)
                                for context, prediction in zip(contexts, prediction_texts)]
            spans_pred.extend(spans_pred_batch)

            batch_metrics = calculate_metrics(
                spans_pred_batch,
                spans_true_batch,
                options=options
            )

            # TODO refactor
            for metric_class, metric_dict in batch_metrics.items():
                writer.add_scalars(metric_class, metric_dict)
                # for metric_name, value in metric_dict.items():
                #     tag = f"{metric_class}_{metric_name}"
                #     writer.add_scalar(tag, value)

            epoch_loss.append(loss.item())
            writer.add_scalar(
                "batch loss / evaluation", loss.item(), epoch * len(dataloader) + i
            )

        epoch_metrics = calculate_metrics(
            spans_pred,
            spans_true,
            options=options
        )

        show_classification_report(epoch_metrics)


def get_sample_text_prediction(
        model: T5ForConditionalGeneration,
        dataloader: torch.utils.data.DataLoader,
        tokenizer: T5Tokenizer,
        generation_kwargs,
        device,
        n=3
):
    model.eval()

    dataset = dataloader.dataset

    ids_to_pick = random.sample(list(range(0, len(dataset))), n)

    for _id in ids_to_pick:
        dataset_item = dataset[_id]

        print(f"Input: {dataset_item.context}")
        print(f"{dataset_item.question}")

        input_ids = tokenizer(
            [dataset_item.context], [dataset_item.question], return_tensors="pt").input_ids

        input_ids = input_ids.to(device)
        answer = model.generate(input_ids, **generation_kwargs)
        answer = tokenizer.decode(answer[0], skip_special_tokens=True)  # TODO change to false

        print(f"Prediction: {answer}")
