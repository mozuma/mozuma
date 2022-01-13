import numpy as np
import pytest
import torch

from mlmodule.contrib.sentences.distilbert.modules import (
    DistilUseBaseMultilingualCasedV2Module,
)
from mlmodule.contrib.sentences.distilbert.transforms import TokenizerTransform
from mlmodule.v2.base.models import MLModuleModelStore


@pytest.mark.parametrize("weights_src", ["provider", "mlmodule"])
def test_embeddings(torch_device: torch.device, weights_src: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        pytest.skip("Skipping as sentence-transformers is not installed")

    sentence = "Hello world"

    st = SentenceTransformer(
        "distiluse-base-multilingual-cased-v2",
        device=torch_device,
    )

    embeddings_provider = st.encode([sentence])

    with torch.no_grad():
        # Loading model from provider
        distilbert = DistilUseBaseMultilingualCasedV2Module(torch.device("cpu"))
        distilbert.eval()
        distilbert.to(torch_device)
        if weights_src == "provider":
            distilbert.set_state_from_provider()
        else:
            MLModuleModelStore().load(distilbert)

        tokenizer = TokenizerTransform(distilbert.get_tokenizer())
        tokens = tokenizer(sentence)

        embedding_mlmodule = (
            distilbert.forward(
                torch.unsqueeze(tokens[0], 0), torch.unsqueeze(tokens[1], 0)
            )
            .cpu()
            .numpy()
        )

    np.testing.assert_array_equal(embeddings_provider, embedding_mlmodule)
