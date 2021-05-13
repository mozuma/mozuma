import os
from typing import Tuple, List

import numpy as np
import pytest
from PIL.Image import Image
from torchvision.transforms import Compose

from mlmodule.contrib.rpn import RPN, CosineSimilarityRegionSelector, RegionEncoder, DenseNet161ImageNetEncoder
from mlmodule.torch.data.base import IndexedDataset
from mlmodule.utils import list_files_in_dir
from mlmodule.torch.data.images import convert_to_rgb, get_pil_image_from_file


@pytest.fixture(scope='session')
def rpn():
    base_path = 'src/mlmodule/contrib/rpn'
    config = f'{base_path}/configs/guided_anchoring/ga_rpn_x101_32x4d_fpn_1x_coco.py'
    checkpoint = f'{base_path}/checkpoints/ga_rpn_x101_32x4d_fpn_1x_coco_20200220-c28d1b18.pth'

    # Initialize RPN
    model = RPN(config, device='cuda:0')
    # Load checkpoint
    with open(checkpoint, 'rb') as f:
        model.load(f)
    return model


@pytest.fixture(scope='session')
def region_encoder():
    return DenseNet161ImageNetEncoder(device='cuda:0')


@pytest.fixture(scope='session')
def region_selector():
    return CosineSimilarityRegionSelector(device='cuda:0')


@pytest.fixture(scope='session')
def images() -> Tuple[List[str], List[Image]]:
    base_path = os.path.join("tests", "fixtures", "faces")
    file_names = list_files_in_dir(base_path, allowed_extensions=('jpg',))
    transforms = Compose([
        get_pil_image_from_file,
        convert_to_rgb,
    ])
    return file_names, [transforms(f) for f in file_names]


def test_region_inference(rpn, images):
    indices, imgs = images
    dataset = IndexedDataset[str, np.ndarray, np.ndarray](indices, imgs)

    num_regions = 10
    indices_out, regions = rpn.bulk_inference(dataset, data_loader_options={'batch_size': 1},
                                              regions_per_image=num_regions, min_score=0.0)

    assert set(indices) == set(indices_out)
    assert len(regions) == len(indices_out)
    for r in regions:
        assert len(r) == num_regions


def test_region_encoding(rpn, region_encoder, images):
    indices, imgs = images
    dataset = IndexedDataset[str, np.ndarray, np.ndarray](indices, imgs)

    num_regions = 10
    indices_out, regions = rpn.bulk_inference(dataset, data_loader_options={'batch_size': 1},
                                              regions_per_image=num_regions, min_score=0.0)
    assert len(indices) == len(indices_out) == len(imgs) == len(regions)

    box_dataset = RegionEncoder.prep_encoding(indices, imgs, regions)
    indices_out, regions_encodings = region_encoder.bulk_inference(box_dataset)
    assert len(indices_out) == len(regions_encodings)

    for enc in regions_encodings:
        assert len(enc) == 2208
