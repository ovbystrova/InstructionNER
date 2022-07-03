from typing import Dict, List, Any

from torch.utils.data import Dataset

from src.core.datatypes import Instance, TaskType
from src.formatters.EntityTask import EntityTaskFormatter
from src.formatters.EntityTypeTask import EntityTypeTaskFormatter
from src.formatters.NERTask import NERTaskFormatter


class NERDataset(Dataset):

    def __init__(
        self, 
        data: List[Dict[str, Any]],
        instructions: Dict[str, str],
        options: List[str],
        tasks: List[TaskType] = [
            TaskType.NER,
            TaskType.ENTITY_EXTRACTOR,
            TaskType.ENTITY_TYPING
        ]
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
        options: Dict[str, List[str]],
        tasks: List[TaskType]
    ) -> List[Instance]:

        instances = []

        # TODO Think how to move it from here
        task_to_formatter = {
            TaskType.ENTITY_EXTRACTOR: EntityTaskFormatter,
            TaskType.ENTITY_TYPING: EntityTypeTaskFormatter,
            TaskType.NER: NERTaskFormatter
            }

        for item in data:

            for task in tasks:

                if not task.value in instructions:
                    continue
                
                instance = task_to_formatter[task].format(
                    data=item,
                    instruction=instructions[task.value],
                    options=options
                )

                instances.append(instance)

        return instances