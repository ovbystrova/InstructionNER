from dis import Instruction
from typing import Dict

from src.core.datatypes import Instance, Preffix
from src.formatters import Formatter


class EntityTaskFormatter(Formatter):

    @classmethod
    def format(
        data: Dict[str],
        instruction: str
    ) -> Instance:

        instance = Instance(
            context=Preffix.CONTEXT + data["context"],
            instruction = Preffix.INSTRUCTION + instruction,
            options=None
        )

        return instance