import configparser
import json
import random
from pathlib import Path
from typing import Union

import numpy as np
import torch
import yaml


def set_global_seed(seed: int):
    """
    Set global seed for reproducibility.
    Args:
        seed (int): Seed.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_config(config_file: Union[str, Path]) -> configparser.ConfigParser:
    """
    Load configuration file.
    :param str config_file: location of configuration file.
    :return: config.
    :rtype: configparser.ConfigParser
    """

    config_file = Path(config_file)

    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found at path '{config_file.absolute()}'. "
            "Please, create it if it doesn't exist"
        )

    if config_file.suffix == ".ini":
        config = configparser.ConfigParser()
        config.read(str(config_file), encoding="utf-8")
    elif config_file.suffix in {".yaml", ".yml"}:
        config = yaml.safe_load(open(config_file, encoding="utf-8"))
    else:
        raise NotImplementedError(
            f"Not implemented reading from '{config_file.suffix}' files. "
            f"Current file path is '{config_file.absolute()}'"
        )

    return config


def loads_json(filepath: str):
    """
    Load json file
    :param filepath: str
    :return:
    """
    with open(filepath, "r") as f:
        data = f.read()
        data = json.loads(data)
    return data


def load_json(filepath: str):
    """
    Load json file
    :param filepath: str
    :return:
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    return data
