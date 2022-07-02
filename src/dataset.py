from typing import Dict, List

from torch.utils.data import Dataset

from src.core.datatypes import Instance, DatasetType, Preffix
from src.formatters.EntityTask import EntityTaskFormatter
from src.formatters.EntityTypeTask import EntityTypeTaskFormatter
from src.formatters.NERTask import NERTaskFormatter


class NERDataset(Dataset):

    def __init__(
        self, 
        data: List[Dict[str]],
        instructions: Dict[str, str],
        options: List[str],
    ):
        super().__init__()

        self.instances = self._convert_list_to_instances(
            data=data,
            instructions=instructions,
            options=options
        )

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> Instance:
        return self.instances[index]

    def _convert_list_to_instances(
        self,
        data: List[Dict[str]],
        instructions: Dict[str, str],
        options: Dict[str, List[str]]
    ):
        instances = []

        for item in data:
            # TODO this is for NER task only for now
            # TODO whrite a better Formatter for every type of three tasks
            
            entity_instance = EntityTaskFormatter.format(
                data=item,
                instruction=instructions["EntityTask"],
                options=options
            )
            entity_ner_instance = NERTaskFormatter.format(
                data=item,
                instruction=instructions["NERTask"],
                options=options
            )
            entity_type_instance = EntityTypeTaskFormatter.format(
                data=item,
                instruction=instructions["EntityTypingTask"],
                options=options
            )




            instances.append(entity_instance)

        return instances
