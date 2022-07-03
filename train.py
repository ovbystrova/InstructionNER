import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import T5Tokenizer, T5ForConditionalGeneration

from src.dataset import T5NERDataset
from arg_parse import get_train_args
from utils import set_global_seed, load_config, load_json


if __name__ == "__main__":

    args = get_train_args()

    config = load_config(args.path_to_model_config)

    set_global_seed(int(config["seed"]))

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
    model = T5ForConditionalGeneration.from_pretrained(config["model"]["name"])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["model"]["learning_rate"]),
    )
