import dataclasses
from typing import List


@dataclasses.dataclass
class LabelSet:
    # Must uniquely define this label set
    label_set_unique_id: str
    label_list: List[str]

    def __getitem__(self, item):
        return self.label_list[item]

    def __len__(self):
        return len(self.label_list)
