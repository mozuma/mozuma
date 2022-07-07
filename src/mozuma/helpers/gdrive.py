from io import BytesIO

import requests

_CHUNK_SIZE = 32768


def download_file_from_google_drive(id, chunk_size: int = _CHUNK_SIZE):
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
    for chunk in response.iter_content(chunk_size):
        if chunk:  # filter out keep-alive new chunks
            f.write(chunk)
    f.seek(0)
    return f


def get_confirm_token(response: requests.Response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None
