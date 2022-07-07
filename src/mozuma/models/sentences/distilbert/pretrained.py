from mozuma.models.sentences.distilbert.modules import (
    DistilUseBaseMultilingualCasedV2Module,
)
from mozuma.stores import load_pretrained_model


def torch_distiluse_base_multilingual_v2(
    *args, **kwargs
) -> DistilUseBaseMultilingualCasedV2Module:
    """Multilingual model for semantic similarity

    See [
        distiluse-base-multilingual-cased-v2
    ](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)
    and [sbert documentation](https://www.sbert.net/docs/pretrained_models.html#multi-lingual-models)
    for more information.

    Args:
        device (torch.device, optional): The PyTorch device to initialise the model weights.
            Defaults to `torch.device("cpu")`.
    """
    return load_pretrained_model(
        DistilUseBaseMultilingualCasedV2Module(*args, **kwargs), training_id="cased-v2"
    )
