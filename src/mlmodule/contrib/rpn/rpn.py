import os
import re
from io import BytesIO
from typing import Dict, Tuple, List, Union, TypeVar, Any, Optional

import boto3
import mmcv
import mmcv.parallel as mmcv_parallel
import numpy as np
import torch
from mmcv.runner import load_state_dict as mmcv_load_state_dict
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
from PIL.Image import Image

from mlmodule.box import BBoxPoint, BBoxOutput, BBoxCollection
from mlmodule.contrib.rpn.transforms import RGBToBGR
from mlmodule.torch import BaseTorchMLModule
from mlmodule.torch.data.base import IndexedDataset
from mlmodule.torch.mixins import TorchPretrainedModuleMixin


InputDatasetType = TypeVar('InputDatasetType', bound=IndexedDataset[Any, Any, Union[Image, np.ndarray]])


class RPN(BaseTorchMLModule, TorchPretrainedModuleMixin):
    """  mmdetection Region Proposal Network wrapper """

    def __init__(self, config: Optional[str] = None, device=None):
        super().__init__(device=device)
        # Mirroring the mmdet.aps.inference.init_detector method

        # If no config was passed, load the default config: Guided Anchoring
        if config is None:
            # Get the current directory for this file
            curr_dir = os.path.dirname(__file__)
            config = os.path.join(curr_dir, 'configs', 'guided_anchoring', 'ga_rpn_x101_32x4d_fpn_1x_coco.py')

        # The default state dict to load
        self.state_dict_key = f"pretrained-models/rpn/ga_rpn_x101_32x4d_fpn_1x_coco_20200220-c28d1b18.pth"

        self.model_config = mmcv.Config.fromfile(config)
        self.model_config.model.pretrained = None
        self.model_config.model.train_cfg = None

        # Build the model
        self.model = build_detector(self.model_config.model, test_cfg=self.model_config.get('test_cfg'))

        # Setting up the config to run, as in mmdet/apis/inference inference_detector(.)
        #   -> the data will be a numpy array
        self.cfg = self.model_config.copy()
        self.cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
        self.cfg.data.test.pipeline = replace_ImageToTensor(self.cfg.data.test.pipeline)
        #   -> setting up the `test_pipeline`
        self.pipeline = Compose(self.cfg.data.test.pipeline)

    def eval(self):
        """ Sets the model to eval mode """
        super().eval()
        self.model.eval()

    def get_default_pretrained_state_dict(self: torch.nn.Module, aws_access_key_id: Optional[str] = None,
                                          aws_secret_access_key: Optional[str] = None) -> Dict[str, torch.Tensor]:
        s3 = boto3.resource(
            's3',
            endpoint_url="https://sos-ch-gva-2.exo.io",
            # Optionally using the provided credentials
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        # Select lsir-public-assets bucket
        b = s3.Bucket('lsir-public-assets')

        # Download state dict into BytesIO file
        f = BytesIO()
        b.Object(self.state_dict_key).download_fileobj(f)

        # Load the state dict
        f.seek(0)
        checkpoint = torch.load(f, map_location=lambda storage, loc: storage)

        state_dict = checkpoint['state_dict']
        # Revise keys
        revise_keys = [(r'^module\.', '')]
        for p, r in revise_keys:
            state_dict = {re.sub(p, r, k): v for k, v in state_dict.items()}

        # Don't apply the state_dict to the model, as this is done by mmcv_load_state_dict
        return state_dict

    def load(self, fp=None, pretrained_getter_opts: Dict[str, Any] = None):
        """Loads model from file or from a default pretrained model if `fp=None`

        :param pretrained_getter_opts: Passed to get_default_pretrained_dict
        :param fp:
        :return:
        """
        # Getting state dict
        if fp:
            fp.seek(0)
            state = torch.load(fp)

            state = state['state_dict'] if ('state_dict' in state) else state
            # Revise keys
            revise_keys = [(r'^module\.', '')]
            for p, r in revise_keys:
                state = {re.sub(p, r, k): v for k, v in state.items()}
        else:
            # Getting default pretrained state dict
            state = self.get_default_pretrained_state_dict(**(pretrained_getter_opts or {}))

        # Loading state, as in mmdet.apis.inference.init_detector
        mmcv_load_state_dict(self.model, state, strict=False)
        self.model = self.model.to(self.device)

        return self

    def get_dataset_transforms(self):
        return [
            np.uint8,
            RGBToBGR(),
            lambda x: dict(img=x),
            self.pipeline
        ]

    def forward(self, data):
        results = self.model(return_loss=False, rescale=True, **data)
        boxes = [img_results[:, :4] for img_results in results]
        scores = [img_results[:, 4:] for img_results in results]
        return boxes, scores

    def bulk_inference(self, data: InputDatasetType, data_loader_options=None,
                       regions_per_image=30, min_score=0.7, **opts) -> Tuple[List, List[BBoxCollection]]:

        def collate_custom(batch):
            indices = [x[0] for x in batch]
            data_ = [x[1] for x in batch]
            data_ = mmcv_parallel.collate(data_, samples_per_gpu=len(data_))
            data_['img_metas'] = [img_metas.data[0] for img_metas in data_['img_metas']]
            data_['img'] = [img.data[0] for img in data_['img']]
            return indices, DataContainerWrapper(data_)

        # Force batch size to 1
        data_loader_options = data_loader_options or {}
        data_loader_options['batch_size'] = 1
        data_loader_options['collate_fn'] = collate_custom
        # Set number of regions to choose, minimum score for regions
        result_handler_options = {'num_regions': regions_per_image, 'min_score': min_score}

        return super().bulk_inference(
            data, data_loader_options=data_loader_options, result_handler_options=result_handler_options, **opts
        )

    @classmethod
    def results_handler(
            cls, acc_results: Tuple[List, List[BBoxCollection]],
            new_indices: List,
            new_output: Tuple[List[np.ndarray], List[np.ndarray]],
            num_regions=30,
            min_score=0.7,
    ) -> Tuple[List, List[BBoxCollection]]:
        """ Runs after the forward pass at inference

        :param acc_results: Holds a tuple with indices, list of FacesFeatures namedtuple
        :param new_indices: New indices for the current batch
        :param new_output: New inference output for the current batch
        :param num_regions: The number of regions to keep for each image
        :param min_score: The minimum score a region must have to be kept. Between 0 and 1
        :return:
        """

        # Dealing for the first call where acc_results is None
        output: List[BBoxCollection]
        indices, output = acc_results or ([], [])

        # Converting to list
        new_indices = cls.tensor_to_python_list_safe(new_indices)
        indices += new_indices

        for ind, (boxes, scores) in zip(new_indices, zip(*new_output)):
            # Iterating through output for each image
            img_bbox = []

            if boxes is not None:
                for b, s in zip(boxes[:num_regions], scores[:num_regions]):
                    if s >= min_score:
                        img_bbox.append(BBoxOutput(
                            bounding_box=(BBoxPoint(*b[:2]), BBoxPoint(*b[2:])),  # Extracting two points
                            probability=s.item(),
                            features=None
                        ))
            output.append(img_bbox)

        return indices, output


class DataContainerWrapper:
    """ Wrapper for a dict containing the data, due to the DataContainer class used in mmcv.parallel """

    def __init__(self, data):
        self.data = data

    def to(self, device):
        """ So that the batch.to(device) call in mlmodule.torch.utils's generic_inference(.) works """
        if device != 'cpu':
            return mmcv_parallel.scatter(self.data, [device])[0]
        return self.data
