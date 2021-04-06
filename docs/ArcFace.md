# ArcFace

We are using an implementation of InsightFace in Pytorch (https://github.com/TreB1eN/InsightFace_Pytorch)


## Download the model weights on PC32

```bash
export PUBLIC_ASSETS=/mnt/storage01/lsir-public-assets/pretrained-models
# For text encoders
python -m mlmodule.cli download arcface.ArcFaceFeatures "$PUBLIC_ASSETS/face-detection/model_ir_se50.pt"
```
