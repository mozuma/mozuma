from PIL import Image
from torchvision import transforms

from mlmodule.torch.data.base import BaseIndexedDataset


TORCHVISION_STANDARD_IMAGE_TRANSFORMS = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]


def get_pil_image_from_file(file):
    return Image.open(file)


class ImageDataset(BaseIndexedDataset):
    """
    Dataset returning tuples of item index and PIL image object
    """

    def __init__(self, image_list):
        super().__init__(image_list)
        self.add_transforms([get_pil_image_from_file])
