from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from instruction_ner.core.datatypes import Instance


class InstanceFormatter(ABC):
    @abstractmethod
    def format_instance(
        self,
        context: str,
        entity_values: Optional[Dict[str, List[str]]],
        entity_spans: Optional[List[Tuple[int, int, str]]],
        instruction: str,
        options: List[str],
    ) -> Instance:
        """
        Based on text, values, instruction and list of labels creates Instance objects
        :param context: eg. 'SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .'
        :param entity_values: eg. {'LOC': ['JAPAN'], 'PER': ['CHINA']}
        :param entity_spans: eg. [(9, 15, 'LOC'), (31, 37, 'PER')]
        :param instruction: eg. 'Please extract all entities'
        :param options: eg. ['LOC', 'PER', 'ORG']
        :return: instance object
        """

        raise NotImplementedError
