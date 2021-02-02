from torch.hub import load_state_dict_from_url
import torch.nn as nn

from mlmodule.torch import BaseTorchMLModule
from mlmodule.torch.mixins import TorchPretrainedModuleMixin
from mlmodule.torch.utils import torch_apply_state_to_partial_model, l2_norm
from mlmodule.torch.modules import Bottleneck_IR_SE, get_block

ARCFACE_WEIGHTS_URL = 'https://public.by.files.1drv.com/y4mSzF8WspQJdHylCjlb8rU2gugGzWpYRDk6jdr8FOj1BT4BBA2pugoYt1QUzh9JySvMCizXjYDE2Hhrzqlmkjh1ko15sKBui2YUDdOULaTFOFWjBzfNJ4QpAwO2c5H2V-leZybfFc_kq8EDECiRN19XKSk80BChtag7F-yx5bPdTQQ6c7oHPiCEPjhIbOcsVXAgEa_TF_el9vp3dhk775-4dajBHpI7ve8J4ny8QbKDhc?access_token=EwAAA61DBAAUmcDj0azQ5tf1lkBfAvHLBzXl5ugAAYCvzROMwcERcSmbAU87/cpdk3hs1dPynX7k6KfDDC2/I06sW%2bp17MgTCiyZTbzRJOUdXYZS081%2bXQcnEZkYojLq9Dg4kE39C3x26LZXdXk1IlVTuRg8By1MjJpuSsiGCu0tlU6U6oSv4dp%2bZbc0Htiefh/ZNUmZOv8IhZFNiNvG1/rRbkTMU949mI2aB6djplcftqsCA5MbbnqD8FFu7JvxGyDTJXMCgL1nPvl0Uoqs7Fj3TI/So1oeKV4QVDVwsyXdg5RVz2Fw%2bZC3E21/SR0AYhQYqWoIQR3PJwztIeu74FX4cku0suR16hM8I1VCMkeo0TZ%2bHKacklqeLE2ycCcDZgAACAY0KjjUUyWg0AHqxRa36IgqWwc6e2bd%2bEuvbuvDk6hEXpLWarMrDb17gEcO3GckVr/mzpAap6Y8jbM%2b6y1duEWNijbQRjWiLePn0D5UZ%2byyzJSoV5TMAZKt/jt5%2bN5VVQlTwo13psuPp5EUUHRgXN7Rfmd4sRJh9Hd/SpfgGQG%2bHy2CFciMroMqdIdnZSRZbueofOgGH2AyNK/PQLOmu31e8sBxKiH2OYuhx0IOYexURLiaJKWapUlafjTdGIb7Dyfk9w3BlKZ3XrEsoydlYNWyZoO/fP/fodh9UiSGnWwdTIh6f%2bk6Ux/ezZ1Vrhyng3Kn74EbEUAnDCmvJQO04XXaHNhWMF3ThutzhE6EMQ696ytp52bPeZK2qRZ%2beyRNRBO/SZd9kclWGz5mAwJrQNW9nDSvMoTY/iq2wnAsi7CybVyvrpLBn06jCggJC3dyh2xvtRf229ntxKSK5VvQAHROCPQUMab6yssNuiYcVgv06DRTsgL7seONYnCwMni0TNCSuNqIzMUOtNSJrbKiRCSZueWOFHm1fCxlEMxQ8xXkviufvKY/CTIfH1RdE4PfyR3z62eC2KXFxZZ40KiH0wIHUHwDUpjaBSoAsYqaz0VKiQDiXRll8KbPfhUC'


class BaseArcFaceModule(BaseTorchMLModule, TorchPretrainedModuleMixin):

    def __init__(self, device=None, drop_ratio=0.6):
        super().__init__(device=device)
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64))
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(drop_ratio),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    Bottleneck_IR_SE(bottleneck.in_channel,
                                     bottleneck.depth,
                                     bottleneck.stride))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)

    def get_default_pretrained_state_dict(self):
        """Returns the state dict for a pretrained resnet model
        :return:
        """
        # Getting URL to download model
        url = ARCFACE_WEIGHTS_URL
        # Downloading state dictionary
        pretrained_state_dict = load_state_dict_from_url(
            url, file_name="model_ir_se50.pth")
        # Removing deleted layers from state dict and updating the other with pretrained data
        return torch_apply_state_to_partial_model(self, pretrained_state_dict)
