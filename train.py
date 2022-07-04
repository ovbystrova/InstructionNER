import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import T5Tokenizer, T5ForConditionalGeneration

from arg_parse import get_train_args
from train_utils import train

from src.collator import Collator
from src.dataset import T5NERDataset
from utils import set_global_seed, load_config, load_json


if __name__ == "__main__":

    args = get_train_args()

    config = load_config(args.path_to_model_config)

    set_global_seed(int(config["seed"]))

    writer = None
    if args.log_dir is not None:
        writer = SummaryWriter(log_dir=args.log_dir)

    # load all helper files
    options = load_json(args.path_to_options)
    options = options[config["data"]["dataset"]]
    instructions = load_json(args.path_to_instructions)

    # load data files
    data_train = load_json(config["data"]["train"])["markup"]
    data_valid = load_json(config["data"]["valid"])["markup"]
    data_test = load_json(config["data"]["test"])["markup"]

    # Create Datasets
    train_dataset = T5NERDataset(
        data=data_train,
        instructions=instructions,
        options=options
    )
    valid_dataset = T5NERDataset(
        data=data_valid,
        instructions=instructions,
        options=options
    )
    test_dataset = T5NERDataset(
        data=data_test,
        instructions=instructions,
        options=options
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    tokenizer = T5Tokenizer.from_pretrained(config["model"]["name"])
    tokenizer_kwargs = dict(config["tokenizer"])
    model = T5ForConditionalGeneration.from_pretrained(config["model"]["name"])
    model.to(device)

    if config["replace_labels_with_special_tokens"]:
        # TODO add special tokens to tokenizer and model
        pass

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["model"]["learning_rate"]),
    )

    collator = Collator(
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=int(config["model"]["batch_size"]),
        shuffle=True,
        collate_fn=collator,
    )

    valid_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=int(config["model"]["batch_size"]),
        shuffle=True,
        collate_fn=collator,
    )

    test_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=int(config["model"]["batch_size"]),
        shuffle=True,
        collate_fn=collator,
    )

    train(
        n_epochs=int(config["model"]["n_epoch"]),
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        writer=writer,
        device=device,
    )
