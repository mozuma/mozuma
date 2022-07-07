# Distiluse Multilingual

Text model for computing sentence embeddings in multiple languages based on [Sentence-Transformers](https://www.sbert.net/examples/training/multilingual/README.html) framework[@reimers_2020_multilingual_sentence_bert].

## Pre-trained models

{% for model in models.sentences -%}
::: mozuma.models.sentences.{{ model.factory }}
    rendering:
        show_signature: False
{% endfor %}


## Base model

This model is an implementation of a [`TorchMlModule`][mozuma.torch.modules.TorchMlModule].

::: mozuma.models.sentences.distilbert.modules.DistilUseBaseMultilingualCasedV2Module
    selection:
        members: none


## Pre-trained state origins

See the [stores documentation](../references/stores.md) for usage.

::: mozuma.models.sentences.distilbert.stores.SBERTDistiluseBaseMultilingualCasedV2Store
    selection:
        members: none
