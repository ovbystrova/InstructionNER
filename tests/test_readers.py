import json
from pathlib import Path
from typing import List
from unittest import TestCase

import pandas as pd
from parameterized import parameterized

from instruction_ner.core.datatypes import DatasetField, Span
from instruction_ner.readers import CONLLReader, MITReader, SpacyReader


# TODO optimize these tests: looks like they can be simplified
class TestConllReader(TestCase):
    maxDiff = None

    test_data_dir = Path(__file__).parent / "data" / "readers"

    input_file = test_data_dir / "conll_input.txt"
    output_file = test_data_dir / "conll_output.json"

    with open(input_file, "r") as f:
        input_data = [x.strip("\n") for x in f.readlines()]

    with open(output_file, "r") as f:
        output_data = json.loads(f.read())

        for element in output_data:
            spans = element[DatasetField.ENTITY_SPANS.value]
            element[DatasetField.ENTITY_SPANS.value] = [
                Span.from_json(x) for x in spans
            ]

    @parameterized.expand([(input_data, output_data)])
    def test_conll_reader(self, input_lines: List[str], output_true):

        reader = CONLLReader()

        output_pred = reader.read(data=input_lines)

        self.assertListEqual(output_pred, output_true)

    @parameterized.expand([(input_file.as_posix(), output_data)])
    def test_conll_reader_from_file(self, input_conll_file: str, output_true):
        reader = CONLLReader()

        output_pred = reader.read_from_file(path_to_file=input_conll_file)

        self.assertListEqual(output_pred, output_true)


class TestSpacyReader(TestCase):

    maxDiff = None

    test_data_dir = Path(__file__).parent / "data" / "readers"

    input_file_csv = test_data_dir / "spacy_input.csv"
    input_file_xlsx = test_data_dir / "spacy_input.xlsx"
    output_file = test_data_dir / "spacy_output.json"

    input_data = pd.read_csv(input_file_csv, sep=";")

    with open(output_file, "r") as f:
        output_data = json.loads(f.read())

        for element in output_data:
            spans = element[DatasetField.ENTITY_SPANS.value]
            element[DatasetField.ENTITY_SPANS.value] = [
                Span.from_json(x) for x in spans
            ]

    @parameterized.expand([(input_data, output_data)])
    def test_spacy_reader(self, input_data: pd.DataFrame, output_true):

        reader = SpacyReader()

        output_pred = reader.read(data=input_data)

        self.assertListEqual(output_pred, output_true)

    @parameterized.expand(
        [(input_file_csv, output_data), (input_file_xlsx, output_data)]
    )
    def test_spacy_reader_from_file(self, input_file: str, output_true):
        reader = SpacyReader()

        output_pred = reader.read_from_file(path_to_file=input_file)

        self.assertListEqual(output_pred, output_true)

    def test_spacy_reader_wrong_columns(self):

        df_wrong = pd.DataFrame(columns=["texts", "labels"])
        reader = SpacyReader()

        self.assertRaises(
            ValueError,
            reader.read,
            df_wrong,
        )

    @parameterized.expand(["test.docs"])
    def test_spacy_reader_wrong_file_extension(self, input_filename: str):
        reader = SpacyReader()

        self.assertRaises(ValueError, reader.read_from_file, input_filename)


class TestMITReader(TestCase):
    maxDiff = None

    test_data_dir = Path(__file__).parent / "data" / "readers"

    input_file = test_data_dir / "mit_input.bio"
    output_file = test_data_dir / "mit_output.json"

    with open(input_file, "r") as f:
        input_data = [x.strip("\n") for x in f.readlines()]

    with open(output_file, "r") as f:
        output_data = json.loads(f.read())

        for element in output_data:
            spans = element[DatasetField.ENTITY_SPANS.value]
            element[DatasetField.ENTITY_SPANS.value] = [
                Span.from_json(x) for x in spans
            ]

    @parameterized.expand([(input_data, output_data)])
    def test_mit_reader(self, input_lines: List[str], output_true):

        reader = MITReader()

        output_pred = reader.read(data=input_lines)

        self.assertListEqual(output_pred, output_true)

    @parameterized.expand([(input_file.as_posix(), output_data)])
    def test_mit_reader_from_file(self, input_file: str, output_true):
        reader = MITReader()

        output_pred = reader.read_from_file(path_to_file=input_file)

        self.assertListEqual(output_pred, output_true)
