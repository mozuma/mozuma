import dataclasses
from typing import List


@dataclasses.dataclass(frozen=True)
class LabelSet:
    """Label set is an ordered list of labels used for classification tasks

    Attributes:
        label_set_unique_id (str): Unique identifier for a label set
        label_list (List[str]): Ordered list of labels

    Example:
        LabelSet objects are used as classic lists:

        ```python
        animal_labels = LabelSet(
            label_set_unique_id="animals",
            label_list=["cat", "dog"]
        )
        print(animal_labels.label_set_unique_id)    # "animals"
        print(animal_labels[0])     # "cat"
        print(animal_labels[1])     # "dog"
        print(len(animal_labels))   # 2
        ```
    """

    # Must uniquely define this label set
    label_set_unique_id: str
    label_list: List[str]

    def __getitem__(self, item):
        return self.label_list[item]

    def __len__(self):
        return len(self.label_list)
