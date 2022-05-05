import dataclasses
from typing import Dict, List


@dataclasses.dataclass(frozen=True)
class LabelSet:
    """Label set is an ordered list of labels used for classification tasks

    Attributes:
        label_set_unique_id (str): Unique identifier for a label set
        label_list (List[str]): Ordered list of labels
        label_to_idx (dict): Dict with items (label_name, label_index)

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

        print(animal_labels.get_label_ids(["dog"])) # [1]
        ```
    """

    # Must uniquely define this label set
    label_set_unique_id: str
    label_list: List[str]
    label_to_idx: Dict[str, int] = dataclasses.field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._make_classes()

    def _make_classes(self):
        label_to_idx = {class_: idx for idx, class_ in enumerate(self.label_list)}

        object.__setattr__(self, "label_to_idx", label_to_idx)

    def get_label_ids(self, labels: List[str]) -> List[int]:
        """Get the list of label indices for the provided labels.

        Arguments:
            labels (List[str]): list of labels of which to obtain indices

        Raises:
            ValueError: when a label in `labels` is not found in the label set

        Returns:
            indices (List[int]): list of indices
        """
        targets = []
        for label in labels:
            if label in self.label_list:
                targets.append(self.label_to_idx[label])
            else:
                raise ValueError(f"Label '{label}' is not in the label set")

        return targets

    def __getitem__(self, item):
        return self.label_list[item]

    def __len__(self):
        return len(self.label_list)
