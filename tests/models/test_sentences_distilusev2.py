import numpy as np
import pytest
import torch

from mlmodule.models.sentences import torch_distiluse_base_multilingual_v2
from mlmodule.models.sentences.distilbert.transforms import TokenizerTransform


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
        distilbert = torch_distiluse_base_multilingual_v2(torch.device("cpu"))
        distilbert.eval()
        distilbert.to(torch_device)

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
