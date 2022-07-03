from typing import Dict, List, Any

from src.core.datatypes import Instance, Preffix
from src.core.formatter import Formatter


class NERTaskFormatter(Formatter):
    """
    Task: Given sentence extract entity values and map them with entity labels
    """
    @classmethod
    def format_instance(
            cls,
            data: Dict[str, Any],
            instruction: str,
            options: List[str]
    ) -> Instance:

        instruction = Preffix.INSTRUCTION.value + instruction
        options = Preffix.OPTIONS.value + ", ".join(options)
        question = instruction + " " + options

        instance = Instance(
            context=Preffix.CONTEXT.value + data["context"],
            question=question,
            answer=Formatter.format_answer(data["entities"])
        )

        return instance
