from typing import List

import requests
from torch.hub import load_state_dict_from_url

from mozuma.models.arcface.modules import TorchArcFaceModule
from mozuma.states import StateKey, StateType
from mozuma.stores.list import AbstractListStateStore

# Discovered by looking at OneDrive network activity when downloading a file manually from the Browser
ONE_DRIVE_API_CALL = (
    "https://api.onedrive.com/v1.0/drives/CEC0E1F8F0542A13/items/CEC0E1F8F0542A13!835?"
    "select=id,@content.downloadUrl&authkey=!AOw5TZL8cWlj10I"
)


class ArcFaceStore(AbstractListStateStore[TorchArcFaceModule]):
    """Gets the pretrained state dir from OneDrive

    URL: https://github.com/TreB1eN/InsightFace_Pytorch#2-pretrained-models--performance
    Model: IR-SE50
    """

    @property
    def available_state_keys(self) -> List[StateKey]:
        return [
            StateKey(
                state_type=StateType(backend="pytorch", architecture="arcface"),
                training_id="insightface",
            )
        ]

    def state_downloader(self, model: TorchArcFaceModule, state_key: StateKey) -> None:
        # Getting a download link from OneDrive
        response = requests.get(ONE_DRIVE_API_CALL)
        response.raise_for_status()
        download_url = response.json()["@content.downloadUrl"]

        # Downloading state dict
        pretrained_state_dict = load_state_dict_from_url(
            download_url, file_name="model_ir_se50.pth", map_location=model.device
        )

        model.load_state_dict(pretrained_state_dict)
