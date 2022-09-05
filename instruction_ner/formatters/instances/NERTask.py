from typing import Dict, List, Optional, Tuple

from instruction_ner.core.datatypes import Instance, Preffix, Span
from instruction_ner.formatters.Answer import AnswerFormatter
from instruction_ner.formatters.instances import InstanceFormatter


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
        options: List[str],
    ) -> Instance:

        instruction = Preffix.INSTRUCTION.value + instruction
        options_joined = ", ".join(options)
        options_string = Preffix.OPTIONS.value + options_joined
        question = instruction + " " + options_string

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
            entity_values=entity_values,
            entity_spans=entity_spans,
        )

        return instance
