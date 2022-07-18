from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union


class Reader(ABC):
    """
    Abstract class for Reading different datasets
    """
    @abstractmethod
    def read(self, data: Any):
        raise NotImplementedError

    @abstractmethod
    def read_from_file(self, path_to_file: Union[str, Path]):
        raise NotImplementedError
