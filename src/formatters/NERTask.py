from dis import Instruction
from typing import Dict, List

from src.core.datatypes import Instance, Preffix
from src.formatters import Formatter


class NERTaskFormatter(Formatter):

    @classmethod
    def format(
        data: Dict[str],
        instruction: str,
        options: List[str]
    ) -> Instance:

        instance = Instance(
            context=Preffix.CONTEXT + data["context"],
            instruction = Preffix.INSTRUCTION + instruction,
            options=Preffix.OPTIONS +  ", ".join(options)
        )

        return instance