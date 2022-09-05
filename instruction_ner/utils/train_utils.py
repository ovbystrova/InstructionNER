from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from instruction_ner.utils.evaluate_utils import (
    evaluate,
    get_sample_text_prediction,
    update_best_checkpoint,
)


def train(
    n_epochs: int,
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    writer: Optional[SummaryWriter],
    device: torch.device,
    eval_every_n_batches: int,
    pred_every_n_batches: int,
    generation_kwargs: Dict[str, Any],
    options: List[str],
    path_to_save_model: Optional[str],
    metric_name_to_choose_best: str = "f1-score",
    metric_avg_to_choose_best: str = "weighted",
) -> None:

    metrics_best: Dict[str, Dict[str, float]] = {}

    for epoch in range(n_epochs):
        print(f"Epoch [{epoch + 1} / {n_epochs}]\n")

        metrics_best = train_epoch(
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
            options=options,
            path_to_save_model=path_to_save_model,
            metrics_best=metrics_best,
            metric_name_to_choose_best=metric_name_to_choose_best,
            metric_avg_to_choose_best=metric_avg_to_choose_best,
        )

        evaluate_metrics = evaluate(
            model=model,
            tokenizer=tokenizer,
            dataloader=test_dataloader,
            writer=writer,
            device=device,
            epoch=epoch,
            generation_kwargs=generation_kwargs,
            options=options,
        )

        if path_to_save_model is None:
            continue

        metrics_best = update_best_checkpoint(
            metrics_best=metrics_best,
            metrics_new=evaluate_metrics,
            metric_name=metric_name_to_choose_best,
            metric_avg=metric_avg_to_choose_best,
            model=model,
            tokenizer=tokenizer,
            path_to_save_model=path_to_save_model,
        )


def train_epoch(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    writer: Optional[SummaryWriter],
    device: torch.device,
    epoch: int,
    eval_every_n_batches: int,
    pred_every_n_batches: int,
    generation_kwargs: Dict[str, Any],
    options: List[str],
    path_to_save_model: Optional[str],
    metrics_best: Dict[str, Dict[str, float]],
    metric_name_to_choose_best: str = "f1-score",
    metric_avg_to_choose_best: str = "weighted",
    test_dataloader: torch.utils.data.DataLoader = None,
) -> Dict[str, Dict[str, float]]:
    """
    One training cycle (loop).
    Args:
        :param metric_avg_to_choose_best:
        :param metric_name_to_choose_best:
        :param metrics_best:
        :param path_to_save_model:
        :param options: list of labels in dataset
        :param generation_kwargs: arguments for generation (ex., beam_size)
        :param test_dataloader:
        :param train_dataloader:
        :param pred_every_n_batches: do sample prediction every n batches
        :param eval_every_n_batches: do evaluation every n batches
        :param model:
        :param optimizer:
        :param writer: tensorboard writer (optional)
        :param epoch: current epoch
        :param device: cpu or cuda
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
        if writer:
            writer.add_scalar(
                "batch loss / train", loss.item(), epoch * len(train_dataloader) + i
            )

        if i % eval_every_n_batches == 0 and i >= eval_every_n_batches:
            if test_dataloader is not None:
                evaluate_metrics = evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    dataloader=test_dataloader,
                    writer=writer,
                    device=device,
                    epoch=epoch,
                    generation_kwargs=generation_kwargs,
                    options=options,
                )

                metrics_best = update_best_checkpoint(
                    metrics_best=metrics_best,
                    metrics_new=evaluate_metrics,
                    metric_name=metric_name_to_choose_best,
                    metric_avg=metric_avg_to_choose_best,
                    model=model,
                    tokenizer=tokenizer,
                    path_to_save_model=path_to_save_model,
                )

        if i % pred_every_n_batches == 0 and i >= pred_every_n_batches:
            get_sample_text_prediction(
                model=model,
                tokenizer=tokenizer,
                dataloader=test_dataloader,
                device=device,
                generation_kwargs=generation_kwargs,
                options=options,
            )

    avg_loss = np.mean(epoch_loss)
    print(f"Train loss: {avg_loss}\n")
    if writer:
        writer.add_scalar("loss / train", avg_loss, epoch)

    return metrics_best
