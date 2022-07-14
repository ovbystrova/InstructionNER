from typing import Dict, List, Tuple, Optional

from src.core.datatypes import Instance, Preffix
from src.formatters.instances import InstanceFormatter


class EntityExtractTaskFormatter(InstanceFormatter):
    """
    Task: Extract all entity values from the text without their labels
    """
    def format_instance(
            self,
            context: str,
            entity_values: Optional[Dict[str, List[str]]],
            entity_spans: Optional[List[Tuple[int, int, str]]],
            instruction: str,
            options: List[str]
    ) -> Instance:

        question = Preffix.INSTRUCTION.value + instruction

        answer = None

        if entity_values is not None:
            answers = []
            for entity_values in entity_values.values():
                answers.extend(entity_values)
            answer = ", ".join(answers) + "."

        instance = Instance(
            context=Preffix.CONTEXT.value + context,
            question=question,
            answer=answer,
            entity_spans=entity_spans,
            entity_values=entity_values
        )

        return instance
