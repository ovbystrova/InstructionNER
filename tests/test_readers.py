import json
from pathlib import Path
from typing import List
from unittest import TestCase
from parameterized import parameterized

from src.core.datatypes import Span, DatasetField
from src.readers import CONLLReader


class TestReader(TestCase):
    maxDiff = None

    test_data_dir = Path(__file__).parent / "data" / "readers"

    input_conll_file = test_data_dir / "conll_input.txt"
    output_conll_file = test_data_dir / "conll_output.json"

    with open(input_conll_file, "r") as f:
        input_conll_data = [x.strip("\n") for x in f.readlines()]

    with open(output_conll_file, "r") as f:
        output_conll_data = json.loads(f.read())

        for element in output_conll_data:
            spans = element[DatasetField.ENTITY_SPANS.value]
            element[DatasetField.ENTITY_SPANS.value] = [Span.from_json(x) for x in spans]

    @parameterized.expand([
        (input_conll_data, output_conll_data)
    ])
    def test_conll_reader(self, input_lines: List[str], output_true):

        reader = CONLLReader()

        output_pred = reader.read(
            data=input_lines
        )

        self.assertListEqual(
            output_pred,
            output_true
        )

    @parameterized.expand([
        (input_conll_file.as_posix(), output_conll_data)
    ])
    def test_conll_reader_from_file(self, input_conll_file: str, output_true):
        reader = CONLLReader()

        output_pred = reader.read_from_file(
            path_to_conll_file=input_conll_file
        )

        self.assertListEqual(
            output_pred,
            output_true
        )
