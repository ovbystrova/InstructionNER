from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Union, List, Dict

from instruction_ner.core.datatypes import DatasetField


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

    @staticmethod
    def save_to_json(data: List[Dict[str, Any]], path: Union[str, Path]):
        """
        Save processed data to json file
        :param data: List of sentences with entities
        :param path: where to save the file
        :return:
        """

        if not isinstance(path, str):
            path = str(path)

        for item in data:

            spans = item[DatasetField.ENTITY_SPANS.value]

            spans_json = [span.to_json() for span in spans]
            item[DatasetField.ENTITY_SPANS.value] = spans_json

        with open(path, "w", encoding="utf-8") as f:
            json_string = json.dumps(data, ensure_ascii=False, indent=4)
            f.write(json_string)

        print(f"Saved to {path}")
