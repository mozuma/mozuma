import numpy as np
import pytest
import torch

from mozuma.models.sentences import torch_distiluse_base_multilingual_v2
from mozuma.models.sentences.distilbert.transforms import TokenizerTransform


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

        embedding_mozuma = (
            distilbert.forward(
                (
                    torch.LongTensor(torch.unsqueeze(tokens[0], 0)),
                    torch.FloatTensor(torch.unsqueeze(tokens[1], 0)),
                )
            )
            .cpu()
            .numpy()
        )

    np.testing.assert_array_equal(embeddings_provider, embedding_mozuma)


def test_distiluse_enable_truncation():
    """Test that if truncation is enabled, the model can process long text"""
    # Creating a long text
    long_text = "Hello world " * 1000

    # Getting the model with truncation
    model = torch_distiluse_base_multilingual_v2(
        torch.device("cpu"), enable_tokenizer_truncation=True
    )

    # Encode the text
    encoded = model.get_dataset_transforms()[0](long_text)
    batch_encoded = (encoded[0].unsqueeze(0), encoded[1].unsqueeze(0))

    # Should not raise errors
    model.forward(batch_encoded)
