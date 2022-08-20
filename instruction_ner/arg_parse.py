import argparse


def get_train_args() -> argparse.Namespace:
    """
    Training Argument Parser.
    Returns:
        Namespace: Parsed arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_to_instructions",
        type=str,
        default="instructions.json",
        help="file with instruction prompts",
    )

    parser.add_argument(
        "--path_to_options",
        type=str,
        default="options.json",
        help="file with mapping dataset to its entities",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="runs",
        help="where to log tensorboard",
    )

    parser.add_argument(
        "--eval_every_n_batches",
        type=int,
        default=500,
        help="do evaluation every n batches",
    )

    parser.add_argument(
        "--pred_every_n_batches",
        type=int,
        default=500,
        help="write random sample sample predictions every n batches",
    )

    parser.add_argument(
        "--path_to_model_config",
        type=str,
        required=True,
        default="configs/config.yaml",
        help="path to all necessary information for model",
    )

    parser.add_argument(
        "--path_to_model_save",
        type=str,
        default="runs/model/",
        help="where to save model",
    )

    args = parser.parse_args()

    return args


def get_data_args() -> argparse.Namespace:
    """
    Reader Argument Parser.
    Returns:
        Namespace: Parsed arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_to_file",
        type=str,
        required=True,
        help="path to initial raw data file",
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        choices=["conll2003", "spacy", "mit"],
        help="dataset type to map it with relevant Reader",
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        help="where to save converted dataset",
    )

    args = parser.parse_args()

    return args


def get_evaluate_args() -> argparse.Namespace:
    """
    Evaluation Argument Parser.
    Returns:
        Namespace: Parsed arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path_or_name",
        type=str,
        required=True,
        default="olgaduchovny/t5-base-qa-ner-conll",
        help="path to trained model or HF model name",
    )

    parser.add_argument(
        "--path_to_model_config",
        type=str,
        required=True,
        default="configs/config.yaml",
        help="path to all necessary information for model",
    )

    parser.add_argument(
        "--path_to_options",
        type=str,
        default="options.json",
        help="file with mapping dataset to its entities",
    )

    parser.add_argument(
        "--path_to_instructions",
        type=str,
        default="instructions.json",
        help="file with instruction prompts",
    )

    args = parser.parse_args()

    return args
