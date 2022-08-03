from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from utils.evaluate_utils import evaluate, get_sample_text_prediction


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
        writer: Optional[SummaryWriter],
        device: torch.device,
        epoch: int,
        eval_every_n_batches: int,
        pred_every_n_batches: int,
        generation_kwargs: Dict[str, Any],
        options: List[str],
        test_dataloader: torch.utils.data.DataLoader = None,
) -> None:
    """
    One training cycle (loop).
    Args:
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
                generation_kwargs=generation_kwargs,
                options=options
            )

    avg_loss = np.mean(epoch_loss)
    print(f"Train loss: {avg_loss}\n")
    if writer:
        writer.add_scalar("loss / train", avg_loss, epoch)
