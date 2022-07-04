from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer


# TODO: add metrics calculation
def train(
        n_epochs: int,
        model: T5ForConditionalGeneration,
        tokenizer,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        writer: SummaryWriter,
        device: torch.device,
        eval_every_n_batches,
        pred_every_n_batches
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
            pred_every_n_batches=pred_every_n_batches
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
        test_dataloader: torch.utils.data.DataLoader = None,
) -> None:
    """
    One training cycle (loop).
    Args:
        :param test_dataloader:
        :param train_dataloader:
        :param pred_every_n_batches:
        :param eval_every_n_batches:
        :param model:
        :param optimizer:
        :param writer:
        :param epoch:
        :param device:
        :param tokenizer:
    """

    model.train()

    epoch_loss = []

    for i, inputs in tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc="loop over train batches",
    ):
        optimizer.zero_grad()

        inputs.pop("instances")
        answers = inputs.pop("answers")

        # replace padding token id's of the labels by -100 so it's ignored by the loss
        answers = torch.tensor(answers.input_ids)
        answers[answers == tokenizer.pad_token_id] = -100

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
                    epoch=epoch
                )
        if i % pred_every_n_batches == 0 and i >= pred_every_n_batches:
            pass  # TODO add prediction sample

    avg_loss = np.mean(epoch_loss)
    print(f"Train loss: {avg_loss}\n")
    writer.add_scalar("loss / train", avg_loss, epoch)


def evaluate(
        model: T5ForConditionalGeneration,
        tokenizer: T5Tokenizer,
        dataloader: torch.utils.data.DataLoader,
        writer: SummaryWriter,
        device: torch.device,
        epoch: int,
):
    model.eval()

    epoch_loss = []

    for i, inputs in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="loop over test batches",
    ):

        inputs.pop("instances")
        answers = inputs.pop("answers")

        # replace padding token id's of the labels by -100 so it's ignored by the loss
        answers = torch.tensor(answers.input_ids)
        answers[answers == tokenizer.pad_token_id] = -100

        outputs = model(**inputs, labels=answers)
        loss = outputs.loss

        epoch_loss.append(loss.item())
        writer.add_scalar(
            "batch loss / evaluation", loss.item(), epoch * len(dataloader) + i
        )
