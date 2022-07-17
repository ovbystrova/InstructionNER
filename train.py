import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import T5Tokenizer, T5ForConditionalGeneration

from arg_parse import get_train_args
from train_utils import train

from src.collator import Collator
from src.dataset import T5NERDataset
from src.utils import set_global_seed, load_config, load_json, loads_json

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    args = get_train_args()

    config = load_config(args.path_to_model_config)

    set_global_seed(config["seed"])

    writer = None
    if args.log_dir is not None:
        log_dir = args.log_dir + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir=log_dir)

    # load all helper files
    options = load_json(args.path_to_options)
    options = options[config["data"]["dataset"]]
    instructions = load_json(args.path_to_instructions)

    # load data files
    data_train = loads_json(config["data"]["train"])
    data_valid = loads_json(config["data"]["valid"])
    data_test = loads_json(config["data"]["test"])

    # Create Datasets
    train_dataset = T5NERDataset(
        data=data_train,
        instructions=instructions["train"],
        options=options
    )
    valid_dataset = T5NERDataset(
        data=data_valid,
        instructions=instructions["test"],
        options=options
    )
    test_dataset = T5NERDataset(
        data=data_test,
        instructions=instructions["test"],
        options=options
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load model
    tokenizer = T5Tokenizer.from_pretrained(config["model"]["name"])
    model = T5ForConditionalGeneration.from_pretrained(config["model"]["name"])
    model.to(device)

    tokenizer_kwargs = dict(config["tokenizer"])
    generation_kwargs = dict(config["generation"])

    if config["replace_labels_with_special_tokens"]:
        # TODO add special tokens to tokenizer and model
        pass

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
    )

    collator = Collator(
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        collate_fn=collator,
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        collate_fn=collator,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        collate_fn=collator,
    )

    eval_every_n_batches = args.eval_every_n_batches
    pred_every_n_batches = args.pred_every_n_batches

    train(
        n_epochs=int(config["training"]["n_epoch"]),
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        writer=writer,
        device=device,
        eval_every_n_batches=eval_every_n_batches,
        pred_every_n_batches=pred_every_n_batches,
        generation_kwargs=generation_kwargs,
        options=options
    )
