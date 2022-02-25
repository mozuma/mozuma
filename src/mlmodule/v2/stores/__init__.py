import os

from mlmodule.v2.stores.s3 import S3StateStore


def Store() -> S3StateStore:
    """MlModule model state store.

    Example:
        The store can be used to list available pre-trained states for a model

        ```python
        store = Store()
        states = store.get_state_keys(model.state_type)
        ```

        And load a given state to a model

        ```python
        store.load(model, state_key=states[0])
        ```
    """
    return S3StateStore(
        bucket="lsir-public-assets",
        session_kwargs=dict(
            aws_access_key_id=os.environ.get("MLMODULE_AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("MLMODULE_AWS_SECRET_ACCESS_KEY"),
            profile_name=os.environ.get("MLMODULE_AWS_PROFILE_NAME"),
        ),
        s3_endpoint_url="https://sos-ch-gva-2.exo.io",
        base_path="pretrained-models/",
    )
