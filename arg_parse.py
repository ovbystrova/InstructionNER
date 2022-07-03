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
        help="file with mapping between dataset and NER labels",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        help="file with mapping between dataset and NER labels",
    )

    parser.add_argument(
        "--path_to_model_config",
        type=str,
        required=True,
        default="config.yaml",
        help="path to all necessary information for model",
    )

    args = parser.parse_args()

    return args
