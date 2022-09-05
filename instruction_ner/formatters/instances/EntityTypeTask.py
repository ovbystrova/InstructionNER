from typing import Dict, List, Optional, Tuple

from instruction_ner.core.datatypes import Instance, Preffix, Span
from instruction_ner.formatters.Answer import AnswerFormatter
from instruction_ner.formatters.instances import InstanceFormatter


class EntityTypeTaskFormatter(InstanceFormatter):
    """
    Task: Given sentence and entity values map them with entity labels
    """

    def format_instance(
        self,
        context: str,
        entity_values: Optional[Dict[str, List[str]]],
        entity_spans: Optional[List[Tuple[int, int, str]]],
        instruction: str,
        options: List[str],
    ) -> Instance:

        entity_values_total = None
        if entity_values is not None:
            entity_values_total = []
            for values in entity_values.values():
                entity_values_total.extend(values)

        instruction = (
            Preffix.INSTRUCTION.value
            + instruction
            + ": "
            + ", ".join(entity_values_total)
        )
        options_str = Preffix.OPTIONS.value + ", ".join(options)
        question = instruction + " " + options_str

        if entity_spans is not None:
            entity_spans = [
                Span.from_json(span)
                for span in entity_spans
                if not isinstance(span, Span)
            ]

        instance = Instance(
            context=Preffix.CONTEXT.value + context,
            question=question,
            answer=AnswerFormatter.from_values(entity_values),
            entity_spans=entity_spans,
            entity_values=entity_values,
        )

        return instance
