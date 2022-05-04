# Stores

Stores are used to load and save models state.
They define 3 methods: `get_state_keys`, `load` and `save`.

The `get_state_keys` method allows to list available pre-trained states for a given type of state.
It usually called from a `model`.

```python
# List available model states in store for the model state type
state_keys = store.get_state_keys(model.state_type)

print(state_keys)
# Prints for a ResNet pre-trained on ImageNet:
# [
#   StateKey(
#       state_type=StateType(
#           backend='pytorch',
#           architecture='resnet18',
#           extra=('cls1000',)
#       ),
#       training_id='imagenet'
#   )
# ]
```

See [states](states.md) documentation for more information on how pre-trained states are identified.

Then, a state can be loaded into the `model`.

```python
# Getting the first state key
resnet_imagenet_state_key = state_keys[0]
# Loading it into the model
store.load(model, state_key=resnet_imagenet_state_key)
```

A model can be saved by specifying a `training_id` which should uniquely identify the training activity that yielded this model's state.

```python
store.save(model, training_id="2022-01-01-finetuned-imagenet")
```

See [AbstractStateStore][mlmodule.v2.stores.abstract.AbstractStateStore] for more details on these methods.

## MLModule pre-trained models

MLModule provides model weights for all defined [models](../models/index.md)
through the MlModule [`Store`][mlmodule.v2.stores.Store].

::: mlmodule.v2.stores.Store
    selection:
        members: none

## Alternative stores

These stores can be used if you want to store model states locally or on S3 storage.

::: mlmodule.v2.stores.local.LocalStateStore
    selection:
        members: none


::: mlmodule.v2.stores.s3.S3StateStore
    selection:
        members: none

::: mlmodule.v2.stores.github.GitHUBReleaseStore
    selection:
        members: none


## Write your own store

A store should inherit [`AbstractStateStore`][mlmodule.v2.stores.abstract.AbstractStateStore]
and implement the `save`, `load` and `get_state_keys` methods.

::: mlmodule.v2.stores.abstract.AbstractStateStore

For stores used to download states of a single model, it can be useful to subclass the
[`AbstractListStateStore`][mlmodule.v2.stores.list.AbstractListStateStore] directly.
This makes it easier to define a store from a fix set of states as it is often the case when
integrating the weights from external sources (pre-trained states for a paper, hugging face...).
See [`SBERTDistiluseBaseMultilingualCasedV2Store`][mlmodule.contrib.sentences.distilbert.stores.SBERTDistiluseBaseMultilingualCasedV2Store]
for an example.

::: mlmodule.v2.stores.list.AbstractListStateStore
