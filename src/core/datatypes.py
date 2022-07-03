from enum import Enum

from dataclasses import dataclass


class DatasetType(Enum):
    CONLL2003 = "conll2003"
    AITS = "atis"
    MIT_MOVIE = "mit_movie"
    MIT_RESTAURANT = "mit_restaurant"


class Preffix(Enum):
    CONTEXT = "Sentence: "
    INSTRUCTION = "Instruction: "
    OPTIONS = "Options: "


class TaskType(Enum):
    ENTITY_EXTRACTOR = "EntityExtractor"
    ENTITY_TYPING = "EntityTyping"
    NER = "NER"


@dataclass
class Instance:
    context: str
    question: str
    answer: str
