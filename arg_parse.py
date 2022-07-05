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
        "--eval_every_n_batches",
        type=int,
        default=200,
        help="do evaluation every n batches",
    )

    parser.add_argument(
        "--pred_every_n_batches",
        type=int,
        default=200,
        help="write one sample prediction every n batches",
    )

    parser.add_argument(
        "--path_to_model_config",
        type=str,
        required=True,
        default="config.yaml",
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
