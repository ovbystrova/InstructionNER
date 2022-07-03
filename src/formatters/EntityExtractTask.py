from typing import Dict, List, Any

from src.core.datatypes import Instance, Preffix
from src.core.formatter import Formatter


class EntityExtractTaskFormatter(Formatter):
    """
    Task: Extract all entity values from the text without their labels
    """

    @classmethod
    def format_instance(
            cls,
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
