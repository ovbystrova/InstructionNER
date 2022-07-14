from enum import Enum
from typing import Dict, List, Optional, Tuple

from dataclasses import dataclass


@dataclass()
class Instance:
    """
    Core Instance dataclass.
    :param context (str): initial text
    :param question (str): question for QA model
    :param answer (optional): raw answer from QA model
    :param spans (optional): List of Spans
    """
    context: str
    question: str
    answer: Optional[str]
    entity_spans: Optional[List[Tuple[int, int, str]]]
    entity_values: Optional[Dict[str, List[str]]]


@dataclass()
class Span:
    """
    Core Span dataclass
    :param start(int): start index of an entity
    :param end(int): end index of an entity
    :param label(str): entity label
    """
    start: int
    end: int
    label: str

    def to_json(self):
        return {"start": self.start, "end": self.end, "label": self.label}

    @staticmethod
    def from_json(data):
        return Span(
            start=int(data["start"]),
            end=int(data["end"]),
            label=data["label"]
        )


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


class DatasetField(Enum):
    CONTEXT = "context"
    ENTITY_VALUES = "entity_values"
    ENTITY_SPANS = "entity_spans"


class Language(Enum):
    EN = "en"
