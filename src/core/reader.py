from abc import ABC, abstractmethod
from typing import Any


class Reader(ABC):
    """
    Abstract class for Reading different datasets
    """
    @abstractmethod
    def read(self, data: Any):
        raise NotImplementedError
