
States are managed with two concepts:

* [`StateType`][mlmodule.states.StateType]: Represents a family of states that are
    [compatible with each other][mlmodule.states.StateType.is_compatible_with].
    In general, a model can be loaded with any pre-trained state if it matches its `state_type` attribute.
* [`StateKey`][mlmodule.states.StateKey]: The identifier of a state instance,
    it should uniquely identify the result of a training activity for a model.

::: mlmodule.states.StateType

::: mlmodule.states.StateKey

::: mlmodule.states.VALID_NAMES
