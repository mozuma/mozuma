# Datasets

## Torch datasets

Torch dataset are implementing dataset as suggested in the
[Datasets & Dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
guide.
These datasets are compatible with `torch.utils.data.DataLoader`.


::: mlmodule.v2.torch.datasets.ListDataset

::: mlmodule.v2.torch.datasets.OpenBinaryFileDataset

::: mlmodule.v2.torch.datasets.OpenImageFileDataset

## Write your own dataset

Dataset types depends on the runner used,
refer to the [runner list](runners.md) to know which type to implement.

Below is the list of dataset protocols that are specifyied by `mlmodule`

::: mlmodule.v2.torch.datasets.TorchDataset
