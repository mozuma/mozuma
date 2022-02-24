
States are managed with two concepts:

* [`StateType`][mlmodule.v2.states.StateType]: Represents a family of states that are
    [compatible with each other][mlmodule.v2.states.StateType.compatible_with].
    In general, a model can be loaded with any pre-trained state if it matches its `state_type` attribute.
* [`StateKey`][mlmodule.v2.states.StateKey]: The identifier of a state instance,
    it should uniquely identify the result of a training activity for a model.

::: mlmodule.v2.states.StateType

::: mlmodule.v2.states.StateKey

::: mlmodule.v2.states.VALID_NAMES
