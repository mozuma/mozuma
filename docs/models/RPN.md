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
