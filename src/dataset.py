from typing import Dict, List

from torch.utils.data import Dataset

from src.core.datatypes import Instance


class NERDataset(Dataset):

    def __init__(self, instances: List[Dict[str]]):
        super().__init__()

        self.instances = self._convert_list_to_instances(
            data=instances
        )

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> Instance:
        return self.instances[index]

    @staticmethod
    def _convert_list_to_instances(data: List[Dict[str]]):
        instances = []

        for item in data:
            instance = Instance(
                context=item["context"],
                entities=item["entities"]
            )
            instances.append(instance)

        return instances
