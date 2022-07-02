from abc import abstractmethod, ABC

from src.core.datatypes import Instance

class Formatter(ABC):

    @abstractmethod
    def format(self, *args) -> Instance:
        raise NotImplementedError