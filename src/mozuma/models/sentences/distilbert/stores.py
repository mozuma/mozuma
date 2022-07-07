from typing import List, OrderedDict

import requests
import torch
from tokenizers import Tokenizer

from mozuma.helpers.torch import state_dict_combine
from mozuma.models.sentences.distilbert.modules import (
    DistilUseBaseMultilingualCasedV2Module,
)
from mozuma.states import StateKey, StateType
from mozuma.stores.list import AbstractListStateStore


class SBERTDistiluseBaseMultilingualCasedV2Store(
    AbstractListStateStore[DistilUseBaseMultilingualCasedV2Module]
):
    """Loads weights from SBERT's hugging face

    See [
        Hugging face's documentation
    ](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2).
    """

    transformer_weights_url = (
        "https://cdn-lfs.huggingface.co"
        "/sentence-transformers/distiluse-base-multilingual-cased-v2"
        "/0ea26561995c7c873e177e6801bb80f36511281d4d96c0f62aea6c19e85ddb7b"
    )
    dense_layer_weights_url = (
        "https://cdn-lfs.huggingface.co"
        "/sentence-transformers/distiluse-base-multilingual-cased-v2"
        "/64fe81485f483cee6c54573686e4117a9e6f32e1579022d3621a1487d5bfea58"
    )
    tokenizer_params_url = (
        "https://huggingface.co"
        "/sentence-transformers/distiluse-base-multilingual-cased-v2/raw/main/tokenizer.json"
    )

    @property
    def available_state_keys(self) -> List[StateKey]:
        return [
            StateKey(
                state_type=StateType(
                    backend="pytorch", architecture="sbert-distiluse-base-multilingual"
                ),
                training_id="cased-v2",
            )
        ]

    def state_downloader(
        self, model: DistilUseBaseMultilingualCasedV2Module, state_key: StateKey
    ) -> None:
        # Loading embeddings and transformer weights
        state: OrderedDict[str, torch.Tensor] = torch.hub.load_state_dict_from_url(
            self.transformer_weights_url, map_location=model.device
        )
        # Loading dense layer weights
        dense_state = torch.hub.load_state_dict_from_url(
            self.dense_layer_weights_url, map_location=model.device
        )
        state.update(state_dict_combine(dense=dense_state))
        # Loading weights
        model.load_state_dict(state)
        # Loading tokenizer
        response = requests.get(self.tokenizer_params_url)
        if response.status_code == 200:
            model._tokenizer = Tokenizer.from_str(response.text)
        else:
            raise ValueError(
                f"Cannot load tokenizer configuration from {self.tokenizer_params_url}"
            )
