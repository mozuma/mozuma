# VinVL

Pre-trained large-scale object-attribute detection (OD) model based on the ResNeXt-152 C4 architecture[@Zhang_2021_CVPR].
The OD model has been firstly trained on much larger amounts of data, combining multiple public object detection datasets, including [COCO](https://cocodataset.org/#home), [OpenImages (OI)](https://storage.googleapis.com/openimages/web/index.html), [Objects365](https://www.objects365.org/overview.html), and [Visual Genome (VG)](https://visualgenome.org/). Then it is fine-tuned on VG dataset alone, since VG is the only dataset with label attributes (see [issue #120](https://github.com/microsoft/Oscar/issues/120#issuecomment-898781183)). It predicts objects from 1594 classes with attributes from 524 classes.
See the [code](https://github.com/pzzhang/VinVL) and the [paper](https://arxiv.org/pdf/2101.00529.pdf) for details.


## Pre-trained models

{% for model in models.vinvl -%}
::: mlmodule.models.vinvl.{{ model.factory }}
    rendering:
        show_signature: False
{% endfor %}


## Base model

The VinVL model is an implementation of a [`TorchMlModule`][mlmodule.torch.modules.TorchMlModule].

::: mlmodule.models.vinvl.modules.TorchVinVLDetectorModule
    selection:
        members: none


## Provider store

See the [stores documentation](../references/stores.md) for usage.

::: mlmodule.models.vinvl.stores.VinVLStore
    selection:
        members: none
