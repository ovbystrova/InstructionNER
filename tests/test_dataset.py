from unittest import TestCase

from parameterized import parameterized

from src.dataset import NERDataset
from src.core.datatypes import Instance, Preffix

# Samle input data
input_data = [{
    "context": "London is a capital of Great Britain",
    "entities": {
        "LOC": ["London", "Great Britain"]
        }
}]
instructions = {"NER": "Extract all entities"}
options = ["ORG", "LOC"]

# Sample Expected Instances
expected_instances = [Instance(
    context=Preffix.CONTEXT.value + "London is a capital of Great Britain",
    question=Preffix.INSTRUCTION.value + "Extract all entities" + " " + Preffix.OPTIONS.value + "ORG, LOC",
    answer="London is a LOC, Great Britain is a LOC"
)]

expected_instances_2 = [Instance(
    context=Preffix.CONTEXT.value + "London is a capital of Great Britain",
    question=Preffix.INSTRUCTION.value + "Extract Entities",
    answer="London, Great Britain"
)]

expected_instances_3 = [Instance(
    context=Preffix.CONTEXT.value + "London is a capital of Great Britain",
    question=Preffix.INSTRUCTION.value + "Type Entities: London, Great Britain" + " " + Preffix.OPTIONS.value + "ORG, LOC",
    answer="London is a LOC, Great Britain is a LOC"
)]


class TestDataset(TestCase):
    maxDiff = None

    @parameterized.expand([
        (input_data, instructions, options, 1)
    ], skip_on_empty=True)
    def test_dataset_length(self, data, instructions, options, length_true):
        dataset = NERDataset(
            data=data,
            instructions=instructions,
            options=options
        )
        self.assertEqual(
            len(dataset),
            length_true
        )

    @parameterized.expand([
        (input_data, instructions, options, 0)
    ])
    def test_dataset_getitem(self, data, instructions, options, idx):
        dataset = NERDataset(
            data=data,
            instructions=instructions,
            options=options
        )
        instance = dataset[idx]
        self.assertEqual(
            expected_instances[idx],
            instance
        ) 

    @parameterized.expand([
        (input_data, instructions, options, expected_instances),
        (input_data, {"EntityExtractor": "Extract Entities"}, options, expected_instances_2),
        (input_data, {"EntityTyping": "Type Entities: "}, options, expected_instances_3)
    ])
    def test_dataset_instances(self, data, instructions, options, instances_true):
        dataset = NERDataset(
            data=data,
            instructions=instructions,
            options=options
        )
        self.assertListEqual(
            dataset.instances,
            instances_true
        )
