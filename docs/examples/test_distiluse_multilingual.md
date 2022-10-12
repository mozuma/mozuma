---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3.7.12
    language: python
    name: python3
---

# Testing multilingual semantic text similarity


<a target="_blank" href="https://github.com/mozuma/mozuma/blob/master/docs/examples/test_distiluse_multilingual.ipynb">
  <img src="https://img.shields.io/static/v1?label=&message=See%20the%20source%20code&color=blue&logo=github&labelColor=black" alt="See the source code"/>
</a>
<a target="_blank" href="https://colab.research.google.com/github/mozuma/mozuma/blob/master/docs/examples/test_distiluse_multilingual.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

```python
from mozuma.torch.runners import TorchInferenceRunner
from mozuma.torch.options import TorchRunnerOptions
from mozuma.models.sentences import torch_distiluse_base_multilingual_v2
from mozuma.torch.datasets import ListDataset
from mozuma.callbacks.memory import CollectFeaturesInMemory

import torch
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="white")
%matplotlib inline
```

1. Create lists of sentences in multiple languages

Random sentences selected from https://tatoeba.org/en/

```python
sentences1 = [
    "We were preparing food.",
    "Ni preparis manĝaĵon.",
    "Wij maakten eten klaar.",
    "Nous préparâmes de la nourriture.",
    "たちは食事の準備をした。",
    "Preparamos comida.",
]
sentences2 = [
    "Anĥizo interpretas la orakolon, kaj konvinkas la Trojanojn, ke temas pri la insulo Kreto, el kiu eliris unu el la unuatempaj fondintoj de Trojo.",
    "Anchise explique l'oracle, et persuade aux Troyens qu'il s'agit de l'île de Crète, d'où est sorti un des anciens fondateurs de Troie.",
    "Anquises interpreta o oráculo e convence os troianos de que se trata da ilha de Creta, da qual saiu um dos antigos fundadores de Troia.",
]
sentences3 = [
    "Mi pensas, ke mi devus foriri, ĉar jam estas malfrue.",
    "Je crois que je devrais partir car il se fait tard.",
    "I think I must be leaving since it is getting late.",
]
```

Define a `ListDataset`

```python
dset = ListDataset(sentences1 + sentences2 + sentences3)
```

2. Extract sentence feature vectors

```python
# define sentence embedding model
torch_device = torch.device("cpu")
model = torch_distiluse_base_multilingual_v2(device=torch_device)

# define callback for collecting features
sentence_features = CollectFeaturesInMemory()

# define the runner
runner = TorchInferenceRunner(
    dataset=dset,
    model=model,
    callbacks=[sentence_features],
    options=TorchRunnerOptions(
        data_loader_options={"batch_size": 1}, device=model.device, tqdm_enabled=True
    ),
)
runner.run()
```

3. Compute sentence similarities

```python
cos_sim = sentence_features.features @ sentence_features.features.T
fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.heatmap(cos_sim, cmap="PuRd", annot=True, fmt=".1")
```

We can see on the heatmap below that similar sentences in multiple languages have high cosine similarity scores, while other sentences have low similarities.
