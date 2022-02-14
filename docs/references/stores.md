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

A store should inherit [`AbstractModelStore`][mlmodule.v2.stores.AbstractModelStore]
and implement the `save` or `load` methods.

::: mlmodule.v2.stores.AbstractModelStore
