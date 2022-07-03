from abc import abstractmethod, ABC
from typing import List, Dict, Any

from src.core.datatypes import Instance


class Formatter(ABC):

    @abstractmethod
    def format_instance(
            self,
            data: Dict[str, Any],
            instruction,
            options
    ) -> Instance:
        """

        :param data: context and entities
        :param instruction: prompt for
        :param options: entity labels presented in data context
        :return: instance object
        """
        raise NotImplementedError

    @classmethod
    def format_answer(cls, entities: Dict[str, List[str]]) -> str:
        """
        Base method to turn dictionary of entities in a string
        :param entities: Dict {entity label : list of entity values}
        :return: All entities in a form of "X is a Y", separated by comma
        """

        answers = []

        for entity_label, values in entities.items():

            if entity_label.lower().startswith("a"):
                answers.extend([f"{value} is an {entity_label}" for value in values])
            else:
                answers.extend([f"{value} is a {entity_label}" for value in values])

        answer = ", ".join(answers)

        return answer
