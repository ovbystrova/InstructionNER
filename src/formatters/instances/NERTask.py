from typing import Dict, List, Tuple, Optional

from src.core.datatypes import Instance, Preffix
from src.formatters.instances import InstanceFormatter
from src.formatters.Answer import AnswerFormatter


class NERTaskFormatter(InstanceFormatter):
    """
    Task: Given sentence extract entity values and map them with entity labels
    """
    def format_instance(
            self,
            context: str,
            entity_values: Optional[Dict[str, List[str]]],
            entity_spans: Optional[List[Tuple[int, int, str]]],
            instruction: str,
            options: List[str]
    ) -> Instance:

        instruction = Preffix.INSTRUCTION.value + instruction
        options = Preffix.OPTIONS.value + ", ".join(options)
        question = instruction + " " + options

        instance = Instance(
            context=Preffix.CONTEXT.value + context,
            question=question,
            answer=AnswerFormatter.from_values(entity_values),
            entity_values=entity_values,
            entity_spans=entity_spans
        )

        return instance
