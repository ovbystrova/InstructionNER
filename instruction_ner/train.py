import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import T5Tokenizer, T5ForConditionalGeneration

from arg_parse import get_train_args
from instruction_ner.collator import Collator
from instruction_ner.dataset import T5NERDataset
from instruction_ner.utils.utils import set_global_seed, load_config, load_json, loads_json
from instruction_ner.utils.train_utils import train

import warnings
warnings.filterwarnings("ignore")


def main():
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

    valid_path = config["data"]["valid"]
    if valid_path is None:
        data_train, data_valid = train_test_split(
            data_train,
            test_size=0.15,
            random_state=config["seed"]
        )
    else:
        data_valid = loads_json(config["data"]["valid"])

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

    eval_every_n_batches = args.eval_every_n_batches
    pred_every_n_batches = args.pred_every_n_batches

    path_to_save_trained_model = Path(args.path_to_model_save)
    path_to_save_trained_model.mkdir(parents=True, exist_ok=True)

    do_save_best_checkpoint = bool(config["training"]["do_save_best_checkpoint"])
    path_to_save_best_checkpoint = None
    if do_save_best_checkpoint:
        path_to_save_best_checkpoint = path_to_save_trained_model / "best"
        path_to_save_best_checkpoint.mkdir(exist_ok=True)

    train(
        n_epochs=int(config["training"]["n_epoch"]),
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        test_dataloader=valid_dataloader,
        optimizer=optimizer,
        writer=writer,
        device=device,
        eval_every_n_batches=eval_every_n_batches,
        pred_every_n_batches=pred_every_n_batches,
        generation_kwargs=generation_kwargs,
        options=options,
        path_to_save_model=path_to_save_best_checkpoint.as_posix(),
        metric_name_to_choose_best=config["training"]["metric_name"],
        metric_avg_to_choose_best=config["training"]["metric_avg"]
    )

    path_to_save_model_last = path_to_save_trained_model / "last"
    path_to_save_model_last.mkdir(exist_ok=True)

    model.save_pretrained(path_to_save_trained_model)
    tokenizer.save_pretrained(path_to_save_trained_model)


if __name__ == "__main__":
    main()
