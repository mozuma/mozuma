import os
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image

CHUNK_SIZE = 32768


def list_files_in_dir(dir_path, allowed_extensions=None):
    """Open all files in a directory and returns the opened objects

    :param dir_path:
    :param mode:
    :param allowed_extensions:
    :return:
    """
    return [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if allowed_extensions is None or any(f.endswith(e) for e in allowed_extensions)
    ]


def download_file_from_google_drive(id):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(
        URL, params={"id": id, "confirm": "t"}, allow_redirects=True, stream=True
    )
    response.raise_for_status()
    # token = get_confirm_token(response)

    # if token:
    #     params = {"id": id, "confirm": token}
    #     response = session.get(URL, params=params, stream=True)
    #     response.raise_for_status()

    f = BytesIO()
    for chunk in response.iter_content(CHUNK_SIZE):
        if chunk:  # filter out keep-alive new chunks
            f.write(chunk)
    f.seek(0)
    return f


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def convert_cv2_image_to_pil(cv2_image_arr: np.ndarray) -> Image.Image:
    """Converts image format from OpenCV2 to PIL

    Makes sure the RGB channels are reordered (see https://stackoverflow.com/a/43234001)
    """
    img = cv2.cvtColor(cv2_image_arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)
