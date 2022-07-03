from typing import Dict, List, Any

from torch.utils.data import Dataset
from tqdm import tqdm

from src.core.datatypes import Instance, TaskType
from src.formatters import (
    EntityExtractTaskFormatter,
    EntityTypeTaskFormatter,
    NERTaskFormatter
)


class T5NERDataset(Dataset):

    def __init__(
            self,
            data: List[Dict[str, Any]],
            instructions: Dict[str, str],
            options: List[str],
            tasks: List[TaskType] = (
                    TaskType.NER,
                    TaskType.ENTITY_EXTRACTOR,
                    TaskType.ENTITY_TYPING
            )
    ):
        super().__init__()

        self.instances = self._convert_list_to_instances(
            data=data,
            instructions=instructions,
            options=options,
            tasks=tasks
        )

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> Instance:
        return self.instances[index]

    def _convert_list_to_instances(
            self,
            data: List[Dict[str, Any]],
            instructions: Dict[str, str],
            options: List[str],
            tasks: List[TaskType]
    ) -> List[Instance]:
        """
        Converts raw data into list of Instance objects
        :param data:
        :param instructions: mapping dictionary from task type to relevant instruction
        :param options: list of labels relevant to the whole dataset
        :param tasks: for what tasks create Instances (each task has its own instruction)
        :return: List of Instance objects
        """

        instances = []

        for item in tqdm(data, desc="Prepare Dataset"):

            instances_per_item = self._convert_item_to_instances(
                data_item=item,
                instructions=instructions,
                options=options,
                tasks=tasks
            )

            instances.extend(instances_per_item)
        return instances

    def _convert_item_to_instances(
            self,
            data_item,
            instructions,
            options,
            tasks
    ):
        """
        Creates all task instances from one element of data
        :param data_item:
        :param instructions: mapping dictionary from task type to relevant instruction
        :param options: list of labels relevant to the whole dataset
        :param tasks: for what tasks create Instances (each task has its own instruction)
        :return: Instance
        """

        instances = []

        # TODO think about this dict and whether it is good for DIP
        task_to_formatter = {
            TaskType.ENTITY_EXTRACTOR: EntityExtractTaskFormatter,
            TaskType.ENTITY_TYPING: EntityTypeTaskFormatter,
            TaskType.NER: NERTaskFormatter
        }

        for task in tasks:

            if task.value not in instructions:
                continue

            instance = task_to_formatter[task].format_instance(
                data=data_item,
                instruction=instructions[task.value],
                options=options
            )

            instances.append(instance)

        return instances
