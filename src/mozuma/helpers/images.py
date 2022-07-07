import cv2
import numpy as np
from PIL import Image


def convert_cv2_image_to_pil(cv2_image_arr: np.ndarray) -> Image.Image:
    """Converts image format from OpenCV2 to PIL

    Makes sure the RGB channels are reordered (see https://stackoverflow.com/a/43234001)
    """
    img = cv2.cvtColor(cv2_image_arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def convert_to_rgb(pil_image: Image.Image) -> Image.Image:
    """PIL convert to RGB function.
    Returns the same image object if the image is already RGB.
    """
    if pil_image.mode == "RGB":
        return pil_image
    return pil_image.convert("RGB")
