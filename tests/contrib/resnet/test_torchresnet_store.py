from mlmodule.contrib.resnet.modules import TorchResNetModule
from mlmodule.contrib.resnet.stores import ResNetTorchVisionStore
from mlmodule.v2.states import StateKey


def test_torchresnet_store_load():
    store = ResNetTorchVisionStore()
    model = TorchResNetModule("resnet18")

    # Initial random state
    random_state = model.get_state()

    # Loading weights
    store.load(model, StateKey(state_type=model.state_type, training_id="imagenet"))
    new_state = model.get_state()

    assert random_state != new_state


def test_torchresnet_store_get_state_keys():
    store = ResNetTorchVisionStore()
    model = TorchResNetModule("resnet18")

    state_keys = store.get_state_keys(model.state_type)
    assert len(state_keys) == 1
    assert state_keys[0].training_id == "imagenet"
    assert state_keys[0].state_type == model.state_type
