import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union

from instruction_ner.core.datatypes import DatasetField, Span


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

    @staticmethod
    def _get_entity_values_from_text(sentence: str, entity_spans: List[Span]):
        """
        Get dict of {label: [values]} from sentence and entity Spans
        :param sentence: text in string format  (eg. 'London is the capital of Great Britain')
        :param entity_spans: List of Span object
        :return:
        """

        entity_values: Dict[str, List[str]] = {}

        for entity in entity_spans:
            start, end, label = entity.start, entity.end, entity.label

            entity_value = sentence[start:end]

            if label not in entity_values:
                entity_values[label] = []

            if entity_value not in entity_values[label]:
                entity_values[label].append(entity_value)

        return entity_values
