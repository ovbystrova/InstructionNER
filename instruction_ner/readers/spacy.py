import ast
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

from instruction_ner.core.datatypes import DatasetField, Span
from instruction_ner.core.reader import Reader


class SpacyReader(Reader):

    text_column = "text"
    entities_column = "labels"

    supported_extensions = [".csv", ".xlsx"]

    def read(
        self, data: pd.DataFrame, text_column=None, entities_column=None
    ) -> List[Dict[str, Any]]:
        """
        Main function of SpacyReader. Based on pd.DataFrame with two specific columns
        create json with text, entity spans and values
        :param data: pd.DataFrame with two columns: text_column, entities_column
        :param text_column: str or None (ex. "text")
        :param entities_column: str or None (ex. "labels")
        :return:  List of Dicts where each element is a text with entities
        """
        text_column = text_column if text_column is not None else self.text_column
        entities_column = (
            entities_column if entities_column is not None else self.entities_column
        )

        for column in [text_column, entities_column]:
            if column not in data.columns:
                raise ValueError(
                    f"Expected dataframe to be with column {column}. Got {data.columns}"
                    f"Either rename your columns to default: {self.text_column, self.entities_column}"
                    f"Or pass your column names as parameters to this function."
                )

        data = self.literal_eval(df=data, columns=[self.entities_column])

        data_processed = []
        texts, entities = (
            data[self.text_column].tolist(),
            data[self.entities_column].tolist(),
        )

        for text, entity_spans in zip(texts, entities):

            entity_spans = [
                Span(start=span[0], end=span[1], label=span[2]) for span in entity_spans
            ]

            entity_values = self._get_entity_values_from_text(text, entity_spans)

            dataset_item = {
                DatasetField.CONTEXT.value: text,
                DatasetField.ENTITY_VALUES.value: entity_values,
                DatasetField.ENTITY_SPANS.value: entity_spans,
            }
            data_processed.append(dataset_item)

        return data_processed

    def read_from_file(
        self, path_to_file: Union[str, Path], sep: str = ";"
    ) -> List[Dict[str, Any]]:
        """
        Wrapper around self.read(). Read "path_to_file" and run self.read()
        :param path_to_file: string or Path
        :param sep: separator for pd.DataFrame (default is ";")
        :return: List of Dicts where each element is a sentence with entities
        """
        if isinstance(path_to_file, str):
            path_to_file = Path(path_to_file)

        if path_to_file.suffix.lower() == ".csv":
            df = pd.read_csv(path_to_file, sep=sep)

        elif path_to_file.suffix.lower() == ".xlsx":
            df = pd.read_excel(path_to_file, engine="openpyxl")

        else:
            raise ValueError(
                f"Expected file to be on of {self.supported_extensions}. Got {path_to_file.suffix}"
            )

        data = self.read(data=df)

        return data

    @staticmethod
    def literal_eval(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Helper function to run ast.literal_eval on specific columns in pd.DataFrame
        :param df: pd.DataFrame
        :param columns:  list of columns names
        :return: pd.DataFrame
        """

        for column in columns:

            df[column] = df[column].apply(ast.literal_eval)

        return df
