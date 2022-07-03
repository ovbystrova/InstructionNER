from abc import abstractmethod, ABC
from typing import List, Dict

from src.core.datatypes import Instance

class Formatter(ABC):

    @abstractmethod
    def format(self, *args) -> Instance:
        raise NotImplementedError

    @classmethod
    def format_answer(entities: Dict[str, List[str]]):

        answers = []

        for entity_label, values in entities.items():

            for value in values:
                if entity_label.lower().startswith("a"):
                    answer = "{value} is an {entity_label}"
                else:
                    answer = "{value} is a {entity_label}"

                answers.append(answer)

        answer = " ".join(answers)

        return answer