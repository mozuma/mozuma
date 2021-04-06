import torch
from facenet_pytorch.models.mtcnn import MTCNN, PNet, RNet, ONet


class MLModuleMTCNN(MTCNN):

    def __init__(
            self, image_size=160, margin=0, min_face_size=20,
            thresholds=(0.6, 0.7, 0.7), factor=0.709, post_process=True,
            select_largest=True, selection_method=None, keep_all=False, device=None,
            pretrained=False
    ):
        super().__init__()

        self.image_size = image_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor
        self.post_process = post_process
        self.select_largest = select_largest
        self.keep_all = keep_all
        self.selection_method = selection_method

        self.pnet = PNet(pretrained=pretrained)
        self.rnet = RNet(pretrained=pretrained)
        self.onet = ONet(pretrained=pretrained)

        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            self.to(device)

        if not self.selection_method:
            self.selection_method = 'largest' if self.select_largest else 'probability'
