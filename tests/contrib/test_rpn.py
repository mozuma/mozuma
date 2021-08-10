import os
from typing import Tuple, List

import mmcv.runner
import numpy as np
import pytest
import torch
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
from mmdet.apis import init_detector, inference_detector
from mmdet.core import get_classes
from PIL.Image import Image
from torchvision.transforms import Compose

from mlmodule.box import BBoxPoint, BBoxOutput, BBoxCollection
from mlmodule.contrib.densenet import DenseNet161ImageNetFeatures
from mlmodule.contrib.rpn import RegionFeatures, RPN, CosineSimilarityRegionSelector, RegionEncoder, \
    DenseNet161ImageNetEncoder, get_absolute_config_path
from mlmodule.contrib.rpn.transforms import RegionCrop, StandardTorchvisionRegionTransforms
from mlmodule.torch.data.base import IndexedDataset
from mlmodule.torch.data.images import convert_to_rgb, get_pil_image_from_file
from mlmodule.utils import list_files_in_dir


CONFIG_PATH = get_absolute_config_path('configs/guided_anchoring/ga_rpn_x101_32x4d_fpn_1x_coco.py')


@pytest.fixture(scope='session')
def rpn(gpu_torch_device):
    """ Load mlmodule RPN """
    # Initialize RPN
    model = RPN(device=gpu_torch_device)
    # Load checkpoint
    model.load_state_dict(model.get_default_pretrained_state_dict_from_provider())
    return model


@pytest.fixture(scope='session')
def region_encoder(gpu_torch_device):
    """ Load mlmodule region encoder """
    densenet = DenseNet161ImageNetEncoder(device=gpu_torch_device)
    densenet.load()
    return densenet


@pytest.fixture(scope='session')
def region_selector(gpu_torch_device):
    """ Load mlmodule region selector """
    return CosineSimilarityRegionSelector(device=gpu_torch_device)


@pytest.fixture(scope='session')
def mmdet_model(rpn: RPN, gpu_torch_device):
    """ Load mmdetection model """
    # Initialize the model
    model = init_detector(CONFIG_PATH, device=gpu_torch_device)

    # Load the state dictionary from s3
    state_dict = load_state_dict_from_url(rpn.MMDET_DOWNLOAD_URL)['state_dict']

    # Apply the state dictionary to the model
    mmcv.runner.load_state_dict(model, state_dict, strict=False)
    model.CLASSES = get_classes('coco')
    return model


@pytest.fixture(scope='session')
def images() -> Tuple[List[str], List[Image]]:
    base_path = os.path.join("tests", "fixtures", "faces")
    file_names = list_files_in_dir(base_path, allowed_extensions=('jpg',))
    transforms = Compose([
        get_pil_image_from_file,
        convert_to_rgb,
    ])
    return file_names, [transforms(f) for f in file_names]


@pytest.fixture(scope='session')
def default_mmdet_encodings(mmdet_model, images) -> Tuple[List[str], List[BBoxCollection]]:
    num_regions = 20
    min_score = 0.0
    urls, _ = images
    boxes = []
    for url in urls:
        results = inference_detector(mmdet_model, url)
        regions = results[:num_regions, :4]
        region_scores = results[:num_regions, 4:]

        img_bboxes = []
        for r, s in zip(regions, region_scores):
            if s >= min_score:
                img_bboxes.append(BBoxOutput(
                    bounding_box=(BBoxPoint(*r[:2]), BBoxPoint(*r[2:])),
                    probability=s.item(),
                    features=None
                ))
        boxes.append(img_bboxes)
    return urls, boxes


def point_distance(point1, point2):
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def max_box_distance(box1, box2):
    p1_1, p2_1 = box1.bounding_box
    p1_2, p2_2 = box2.bounding_box
    return max(point_distance(p1_1, p1_2), point_distance(p2_1, p2_2))


def test_region_inference(rpn, images, default_mmdet_encodings):
    indices, imgs = images
    dataset = IndexedDataset[str, Image, Image](indices, imgs)

    num_regions = 10
    indices_out, regions = rpn.bulk_inference(dataset, data_loader_options={'batch_size': 1},
                                              regions_per_image=num_regions, min_score=0.0)

    assert set(indices) == set(indices_out)
    assert len(regions) == len(indices_out)
    for r in regions:
        assert len(r) == num_regions

    mmdet_urls, mmdet_boxes = default_mmdet_encodings
    for mmdet_url, url in zip(mmdet_urls, indices):
        assert mmdet_url == url

    # NOTE: This test can be a bit difficult, as some tiny changes in values due to floating point errors can change
    # the order of the boxes detected, and the cutoff can mean not all boxes have a match.
    # I have more boxes in the mmdet results to compensate for this a bit.

    # Iterate over images
    for mmdet_box_col, box_col in zip(mmdet_boxes, regions):
        # Max euclidean distance difference: 1 pixel
        max_dist = 1
        # Max probability difference: 1%
        max_proba_dist = 0.01
        unmatched_boxes = 0
        unmatched = []

        # Iterate over boxes found using one method. Find a box in the second collection close to the one in the first
        for box in box_col:
            closest_dist = 100
            proba_dist = 1
            for mmdet_box in mmdet_box_col:
                box_dist = max_box_distance(mmdet_box, box)
                if box_dist < closest_dist:
                    closest_dist = box_dist
                    proba_dist = np.abs(mmdet_box.probability - box.probability)

            # Distance greater than 20 pixels away for a corner lists as no match
            if closest_dist > max_dist or proba_dist > max_proba_dist:
                unmatched.append((round(closest_dist), round(proba_dist, 2), box))
                unmatched_boxes += 1

        if unmatched_boxes > 0:
            for box in box_col:
                print(box)
            print('\nUnmatched')
            for box in unmatched:
                print(box)
            print()
            for box in mmdet_box_col:
                print(box)
        assert unmatched_boxes <= 2


def test_region_encoding(rpn, region_encoder, images, gpu_torch_device):
    indices, imgs = images
    dataset = IndexedDataset[str, np.ndarray, np.ndarray](indices, imgs)

    # Extract proposed regions from image
    num_regions = 10
    indices_out, regions = rpn.bulk_inference(dataset, data_loader_options={'batch_size': 1},
                                              regions_per_image=num_regions, min_score=0.0)

    dataset.transforms = []

    assert len(indices) == len(indices_out) == len(imgs) == len(regions)

    # Compute features for regions
    box_dataset = RegionEncoder.prep_encodings(dataset, regions)
    img_reg_indices, img_reg_encodings = region_encoder.bulk_inference(box_dataset)
    indices, box_collections = RegionEncoder.parse_encodings(img_reg_indices, img_reg_encodings)
    assert len(img_reg_indices) == len(img_reg_encodings)
    assert len(indices) == len(box_collections)

    # Check that the regions have very similar features than when re-extracting them with a densenet
    densenet = DenseNet161ImageNetFeatures(device=gpu_torch_device)
    densenet.load()
    densenet.eval()
    cropper = RegionCrop()
    torchvision_transforms = StandardTorchvisionRegionTransforms()
    for img, img_regions, img_features in zip(imgs, regions, box_collections):
        for r, r_with_features in zip(img_regions, img_features):
            assert r.bounding_box == r_with_features.bounding_box
            assert r.probability == r_with_features.probability

            with torch.no_grad():
                region_img = torchvision_transforms(cropper((img, r))).unsqueeze(dim=0).to(gpu_torch_device)
                features = densenet(region_img).detach().squeeze()
                assert features.shape == r_with_features.features.shape
                sim = F.cosine_similarity(features, torch.Tensor(r_with_features.features).to(gpu_torch_device), dim=0)
                assert sim.item() > 0.99


def test_bbox_selector(rpn, region_encoder, region_selector, images):
    indices, imgs = images
    dataset = IndexedDataset[str, np.ndarray, np.ndarray](indices, imgs)

    # Extract proposed regions from image
    num_regions = 10
    indices_out, regions = rpn.bulk_inference(dataset, data_loader_options={'batch_size': 1},
                                              regions_per_image=num_regions, min_score=0.0)

    dataset.transforms = []

    # Compute features for regions
    box_dataset = RegionEncoder.prep_encodings(dataset, regions)
    img_reg_indices, img_reg_encodings = region_encoder.bulk_inference(box_dataset)
    indices, box_collections = RegionEncoder.parse_encodings(img_reg_indices, img_reg_encodings)

    # Select regions based on cosine similarity
    box_dataset_w_features = IndexedDataset[str, BBoxCollection, BBoxCollection](indices, box_collections)
    _, box_dataset_w_features_selected = region_selector.bulk_inference(box_dataset_w_features)

    assert len(box_dataset_w_features) == len(box_dataset_w_features_selected)
    for i in range(len(box_dataset_w_features)):
        original_idx, original_collection = box_dataset_w_features[i]
        new_idx, new_collection = box_dataset_w_features[i]
        assert original_idx == new_idx
        assert len(original_collection) == len(new_collection)


def test_base(rpn, region_encoder, region_selector, images):
    indices, imgs = images
    dataset = IndexedDataset[str, np.ndarray, np.ndarray](indices, imgs)

    region_features = RegionFeatures(rpn, region_encoder, region_selector)
    _, regions_with_features = region_features.bulk_inference(
        dataset,
        regions_per_image=30,
        min_region_score=0.7
    )

    assert len(regions_with_features) == len(dataset)
