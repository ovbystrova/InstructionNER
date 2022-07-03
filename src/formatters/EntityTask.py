from typing import Dict, List, Any

from src.core.datatypes import Instance, Preffix
from src.formatters import Formatter


class EntityTaskFormatter(Formatter):

    @classmethod
    def format(cls,
               data: Dict[str, Any],
               instruction: str,
               options: List[str]
               ) -> Instance:
        question = Preffix.INSTRUCTION.value + instruction
        answers = []
        for entity_values in data["entities"].values():
            answers.extend(entity_values)

        instance = Instance(
            context=Preffix.CONTEXT.value + data["context"],
            question=question,
            answer=", ".join(answers)
        )

        return instance
