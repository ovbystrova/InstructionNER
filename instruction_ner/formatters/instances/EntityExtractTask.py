from typing import Dict, List, Optional, Tuple

from instruction_ner.core.datatypes import Instance, Preffix, Span
from instruction_ner.formatters.instances import InstanceFormatter


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
        options: List[str],
    ) -> Instance:

        question = Preffix.INSTRUCTION.value + instruction

        answer = None

        if entity_values is not None:
            answers = []
            for _entity_values in entity_values.values():
                answers.extend(_entity_values)
            answer = ", ".join(answers) + "."

        if entity_spans is not None:
            entity_spans = [
                Span.from_json(span)
                for span in entity_spans
                if not isinstance(span, Span)
            ]

        instance = Instance(
            context=Preffix.CONTEXT.value + context,
            question=question,
            answer=answer,
            entity_spans=entity_spans,
            entity_values=entity_values,
        )

        return instance
