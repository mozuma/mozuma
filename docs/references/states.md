
States are managed with two concepts:

* [`StateType`][mozuma.states.StateType]: Represents a family of states that are
    [compatible with each other][mozuma.states.StateType.is_compatible_with].
    In general, a model can be loaded with any pre-trained state if it matches its `state_type` attribute.
* [`StateKey`][mozuma.states.StateKey]: The identifier of a state instance,
    it should uniquely identify the result of a training activity for a model.

::: mozuma.states.StateType

::: mozuma.states.StateKey

::: mozuma.states.VALID_NAMES
