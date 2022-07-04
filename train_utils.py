from collections import defaultdict

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
) -> None:
    for epoch in range(n_epochs):
        print(f"Epoch [{epoch + 1} / {n_epochs}]\n")

        train_epoch(
            model=model,
            tokenizer=tokenizer,
            dataloader=train_dataloader,
            optimizer=optimizer,
            writer=writer,
            device=device,
            epoch=epoch,
        )


def train_epoch(
        model: T5ForConditionalGeneration,
        tokenizer: T5Tokenizer,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        writer: SummaryWriter,
        device: torch.device,
        epoch: int,
) -> None:
    """
    One training cycle (loop).
    Args:
        model (AutoModelForQuestionAnswering): QA model.
        dataloader (torch.utils.data.DataLoader): dataloader.
        optimizer (torch.optim.Optimizer): optimizer.
        writer (SummaryWriter): tensorboard writer.
        device (torch.device): cpu or cuda.
        epoch (int): number of current epochs.
        :param model:
        :param dataloader:
        :param optimizer:
        :param writer:
        :param epoch:
        :param device:
        :param tokenizer:
    """

    model.train()

    epoch_loss = []

    for i, inputs in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
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
            "batch loss / train", loss.item(), epoch * len(dataloader) + i
        )
