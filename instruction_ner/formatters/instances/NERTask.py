from typing import Dict, List, Union, Optional

from instruction_ner.core.datatypes import Instance, Preffix, Span
from instruction_ner.formatters.instances import InstanceFormatter
from instruction_ner.formatters.Answer import AnswerFormatter


class NERTaskFormatter(InstanceFormatter):
    """
    Task: Given sentence extract entity values and map them with entity labels
    """
    def format_instance(
            self,
            context: str,
            entity_values: Optional[Dict[str, List[str]]],
            entity_spans: Optional[List[Dict[str, Union[int, str]]]],
            instruction: str,
            options: List[str]
    ) -> Instance:

        instruction = Preffix.INSTRUCTION.value + instruction
        options = Preffix.OPTIONS.value + ", ".join(options)
        question = instruction + " " + options

        if entity_spans is not None:
            entity_spans = [Span.from_json(span)for span in entity_spans]

        instance = Instance(
            context=Preffix.CONTEXT.value + context,
            question=question,
            answer=AnswerFormatter.from_values(entity_values),
            entity_values=entity_values,
            entity_spans=entity_spans
        )

        return instance
