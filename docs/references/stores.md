# Stores

Stores are used to load and save models state.

## MLModule pre-trained models

MLModule provides model weights for all defined [models](../models/index.md)
through the [`MLModuleModelStore`][mlmodule.v2.stores.MLModuleModelStore].

::: mlmodule.v2.stores.MLModuleModelStore
    selection:
        members:
            - load

## Available stores

::: mlmodule.v2.stores.LocalFileModelStore

## Write your own store

A store should inherit [`AbstractStateStore`][mlmodule.v2.stores.AbstractStateStore]
and implement the `save`, `load` and `get_state_keys` methods.

::: mlmodule.v2.stores.AbstractStateStore
