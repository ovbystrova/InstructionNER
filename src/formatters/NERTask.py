from typing import Dict, List, Any

from src.core.datatypes import Instance, Preffix
from src.formatters import Formatter


class NERTaskFormatter(Formatter):

    @classmethod
    def format(cls,
               data: Dict[str, Any],
               instruction: str,
               options: List[str]
               ) -> Instance:

        instruction = Preffix.INSTRUCTION.value + instruction
        _options = Preffix.OPTIONS.value + ", ".join(options)
        question = instruction + " " + _options

        instance = Instance(
            context=Preffix.CONTEXT.value + data["context"],
            question=question,
            answer=Formatter.format_answer(data["entities"])
        )

        return instance
