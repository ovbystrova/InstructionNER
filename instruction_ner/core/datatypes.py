from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union


@dataclass(frozen=True, eq=True)
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
        return Span(start=int(data["start"]), end=int(data["end"]), label=data["label"])

    @staticmethod
    def from_tuple(data):
        return Span(start=int(data[0]), end=int(data[1]), label=data[2])


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
    entity_spans: Optional[Union[List[Span], List[Tuple[int, int, str]]]]
    entity_values: Optional[Dict[str, List[str]]]

    def __str__(self):

        if self.answer is not None:
            return self.context + " " + self.question + " " + self.answer

        return self.context + " " + self.question


class DatasetType(Enum):
    CONLL2003 = "conll2003"
    AITS = "atis"
    MIT = "mit"
    SPACY = "spacy"


class Preffix(Enum):
    CONTEXT = "Sentence: "
    INSTRUCTION = "Instruction: "
    OPTIONS = "Options: "
    ANSWER = "Answer: "


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
