import numpy as np
import pytest
import torch

from mlmodule.contrib.sentences.distilbert.modules import (
    DistilUseBaseMultilingualCasedV2Module,
)
from mlmodule.contrib.sentences.distilbert.transforms import TokenizerTransform
from mlmodule.v2.states import StateKey
from mlmodule.v2.stores import Store


def test_embeddings(torch_device: torch.device):
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
        Store().load(
            distilbert,
            state_key=StateKey(distilbert.state_type, training_id="cased-v2"),
        )

        tokenizer = TokenizerTransform(distilbert.get_tokenizer())
        tokens = tokenizer(sentence)

        embedding_mlmodule = (
            distilbert.forward(
                (
                    torch.LongTensor(torch.unsqueeze(tokens[0], 0)),
                    torch.FloatTensor(torch.unsqueeze(tokens[1], 0)),
                )
            )
            .cpu()
            .numpy()
        )

    np.testing.assert_array_equal(embeddings_provider, embedding_mlmodule)
