import json
from pathlib import Path
from unittest import TestCase

from parameterized import parameterized

from instruction_ner.core.datatypes import Instance
from instruction_ner.dataset import T5NERDataset


def _make_instance(data):
    instance = Instance(
        context=data["context"],
        question=data["question"],
        answer=data["answer"],
        entity_spans=data["entity_spans"],
        entity_values=data["entity_values"],
    )
    return instance


class TestDataset(TestCase):

    maxDiff = None
    test_data_dir = Path(__file__).parent / "data"
    with open(test_data_dir / "test_case_dataset.json") as f:
        data = json.load(f)

    input_data = data["instance"]
    instructions = data["instructions"]
    options = data["options"]

    instances = [_make_instance(instance) for instance in data["instances"]]

    @parameterized.expand(
        [
            (input_data, instructions, options, 3),
            (input_data, {"NER": instructions["NER"]}, options, 1),
        ]
    )
    def test_dataset_length(self, data, instructions, options, length_true):
        dataset = T5NERDataset(data=data, instructions=instructions, options=options)
        length_pred = len(dataset)
        self.assertEqual(length_pred, length_true)

    @parameterized.expand([(input_data, {"NER": instructions["NER"]}, options, 0)])
    def test_dataset_getitem(self, data, instructions, options, idx):
        dataset = T5NERDataset(data=data, instructions=instructions, options=options)
        instance = dataset[idx]
        self.assertEqual(self.instances[idx], instance)

    @parameterized.expand(
        [
            (input_data, {"NER": instructions["NER"]}, options, [instances[0]]),
            (
                input_data,
                {"EntityExtractor": instructions["EntityExtractor"]},
                options,
                [instances[1]],
            ),
            (
                input_data,
                {"EntityTyping": instructions["EntityTyping"]},
                options,
                [instances[2]],
            ),
        ]
    )
    def test_dataset_instances(self, data, instructions, options, instances_true):
        dataset = T5NERDataset(data=data, instructions=instructions, options=options)

        print(dataset[0].entity_values)
        print(instances_true[0].entity_values)

        self.assertListEqual(dataset.instances, instances_true)
