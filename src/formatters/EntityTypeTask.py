from typing import Dict, List, Any

from src.core.datatypes import Instance, Preffix
from src.formatters import Formatter


class EntityTypeTaskFormatter(Formatter):

    @classmethod
    def format(cls,
               data: Dict[str, Any],
               instruction: str,
               options: List[str]
               ) -> Instance:

        entity_values = []
        for values in data["entities"].values():
            entity_values.extend(values)
        instruction = Preffix.INSTRUCTION.value + instruction + ", ".join(entity_values)
        options = Preffix.OPTIONS.value + ", ".join(options)
        question = instruction + " " + options
        
        instance = Instance(
            context=Preffix.CONTEXT.value + data["context"],
            question=question,
            answer=Formatter.format_answer(data["entities"])
        )

        return instance
