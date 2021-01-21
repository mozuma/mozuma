import PIL
from torchvision import transforms as T

from mlmodule.torch.data.files import FilesDataset


TORCHVISION_STANDARD_IMAGE_TRANSFORMS = [
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]


def get_pil_image_from_file(file):
    return PIL.Image.open(file)


class ImageFilesDatasets(FilesDataset):

    def __init__(self, file_list):
        super().__init__(file_list)
        self.add_transforms([get_pil_image_from_file])
