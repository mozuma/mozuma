# Region Proposal Networks

Region proposal networks (RPNs) can be used to select regions of images that contain interesting information. These
regions can then be encoded, using a DenseNet pretrained on the ImageNet dataset, so that a search tool to look for
specific objects in images.

## Implementation

The RPN module is implemented in 4 main files:
* `base.py`: Regroups the different elements needed to extract and process regions, defined in the other files.
* `rpn.py`: Contains the `BaseTorchMLModule` used to select regions from images
* `encoder.py`: Contains the `BaseTorchMLModule` used to compute the encodings for images
* `selector.py`: Contains a `BaseMlModule` that can be used to filter regions inside of a single that are very similar,
which reduces the amount of region encodings that need to be stored.

## Usage

```python
from mlmodule.contrib.rpn import RegionFeatures
from mlmodule.torch.data.images import ImageDataset
from mlmodule.utils import list_files_in_dir

# We need a list of files
file_list = list_files_in_dir('tests/fixtures/cats_dogs', allowed_extensions='jpg')
dataset = ImageDataset(file_list)

# We instantiate the RegionFeatures model
region_features = RegionFeatures().load()
_, regions_with_features = region_features.bulk_inference(
    dataset,
    regions_per_image=30,
    min_region_score=0.7
)
```
