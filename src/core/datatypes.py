import enum
from typing import List, Optional

from dataclasses import dataclass


class DatasetType(enum):
    CONLL2003 = "conll2003"
    AITS = "atis"
    MIT_MOVIE = "mit_movie"
    MIT_RESTAURANT = "mit_restaurant"


class Preffix(enum):
    CONTEXT = "Sentence: "
    INSTRUCTION = "Instruction: "
    OPTIONS = "Options: "


@dataclass
class Instance:
    context: str
    instruction: str
    options: Optional[List[str]]
