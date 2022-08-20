import torch
from torch.utils.data import DataLoader

from arg_parse import get_evaluate_args
from instruction_ner.model import Model
from instruction_ner.collator import Collator
from instruction_ner.dataset import T5NERDataset
from instruction_ner.utils.utils import set_global_seed, load_config, load_json, loads_json
from utils.evaluate_utils import evaluate

import warnings
warnings.filterwarnings("ignore")


def main():
    args = get_evaluate_args()
    config = load_config(args.path_to_model_config)

    set_global_seed(config["seed"])

    # load all helper files
    options = load_json(args.path_to_options)
    options = options[config["data"]["dataset"]]
    instructions = load_json(args.path_to_instructions)

    data_test = loads_json(config["data"]["test"])

    test_dataset = T5NERDataset(
        data=data_test,
        instructions=instructions["test"],
        options=options
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load model
    model_path_or_name = args.model_path_or_name
    model = Model(
        model_path_or_name=model_path_or_name,
        tokenizer_path_or_name=model_path_or_name
    )
    model.model.to(device)

    tokenizer_kwargs = dict(config["tokenizer"])
    generation_kwargs = dict(config["generation"])

    collator = Collator(
        tokenizer=model.tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=True,
        collate_fn=collator,
    )

    evaluate(
        model=model.model,
        tokenizer=model.tokenizer,
        dataloader=test_dataloader,
        writer=None,
        device=device,
        epoch=0,
        generation_kwargs=generation_kwargs,
        options=options
    )


if __name__ == "__main__":
    main()
