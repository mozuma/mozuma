# Datasets

## Torch datasets

Torch dataset are implementing dataset as suggested in the
[Datasets & Dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
guide.
These datasets are compatible with `torch.utils.data.DataLoader`.

### In-memory list datasets

::: mlmodule.torch.datasets.ListDataset
    rendering:
        heading_level: 4
    selection:
        members: none

::: mlmodule.torch.datasets.ListDatasetIndexed
    rendering:
        heading_level: 4
    selection:
        members: none

### Local files datasets

::: mlmodule.torch.datasets.LocalBinaryFilesDataset
    rendering:
        heading_level: 4
    selection:
        members: none

### Image datasets

::: mlmodule.torch.datasets.ImageDataset
    rendering:
        heading_level: 4
    selection:
        members: none

### Bounding box datasets

::: mlmodule.torch.datasets.ImageBoundingBoxDataset
    rendering:
        heading_level: 4
    selection:
        members: none

### Training datasets

::: mlmodule.torch.datasets.TorchTrainingDataset
    rendering:
        heading_level: 4
    selection:
        members: none

## Write your own dataset

Dataset types depends on the runner used,
refer to the [runner list](runners.md) to know which type to implement.

Below is the list of dataset protocols that are specified by `mlmodule`

::: mlmodule.torch.datasets.TorchDataset
