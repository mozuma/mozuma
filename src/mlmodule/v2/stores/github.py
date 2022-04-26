import dataclasses
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar

import requests
from requests.auth import HTTPBasicAuth

from mlmodule.v2.base.models import ModelWithState
from mlmodule.v2.states import StateKey, StateType
from mlmodule.v2.stores.abstract import AbstractStateStore

_ModelType = TypeVar("_ModelType", bound=ModelWithState)
_JsonType = Dict[str, Any]


def get_github_basic_auth() -> Optional[Tuple[str, str]]:
    basic_auth = os.environ.get("GH_API_BASIC_AUTH")

    # If not defined, return None
    if not basic_auth:
        return None

    # Extract {username}:{gh_personal_token}
    elements = basic_auth.split(":")
    username = elements[0]
    gh_personal_token = ":".join(elements[1:])

    return username, gh_personal_token


def get_github_token() -> Optional[str]:
    return os.environ.get("GH_TOKEN")


def call_github_api(method: str, url: str, **kwargs) -> requests.Response:
    """Authenticated call to GitHUB API"""
    # Resolving authentication method
    basic_auth = get_github_basic_auth()
    gh_token = get_github_token()
    if basic_auth:
        kwargs.setdefault("auth", HTTPBasicAuth(*basic_auth))
    elif gh_token:
        kwargs.setdefault("headers", {})
        kwargs["headers"].setdefault("Authorization", f"Bearer {gh_token}")

    return getattr(requests, method)(url, **kwargs)


def paginate_github_api(method: str, url: str, **kwargs) -> Sequence[_JsonType]:
    results: List[_JsonType] = []
    while True:
        # Calling GitHUB
        response = call_github_api(method, url, **kwargs)
        response.raise_for_status()

        # Accumulating results
        results += response.json()

        # Stopping if no next url
        if "next" not in response.links:
            break

        # Getting next URL
        url = response.links["next"]["url"]

        # Removing querystring if exists
        if "params" in kwargs:
            del kwargs["params"]

    return results


def state_type_to_gh_tag(release_name_prefix: str, state_type: StateType) -> str:
    """Formats a state type to a GitHUB release tag"""
    return f"{release_name_prefix}.{state_type.backend}.{state_type.architecture}"


def gh_asset_name_to_state_key(
    state_type: StateType, asset_name: str
) -> Optional[StateKey]:
    """From a state type and an asset name constructs a state key

    Returns:
        StateKey | None: The constructed state key or None if does not match a state file
    """
    if not asset_name.endswith(".state.gzip"):
        return None

    asset_name_parts = asset_name[:-11].split(".")
    extra = tuple(asset_name_parts[:-1])
    training_id = asset_name_parts[-1]
    return StateKey(
        state_type=dataclasses.replace(state_type, extra=extra),
        training_id=training_id,
    )


@dataclasses.dataclass
class GitHUBReleaseStore(AbstractStateStore[_ModelType]):
    """Store implementation leveraging GitHUB releases

    Model weights are stored as assets in a
    [release](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

    We recommend setting up GitHUB authentication to use the
    `get_state_keys` and the `save` methods.
    These methods are calling the [releases API](https://docs.github.com/en/rest/releases/releases)
    and are limited to 60 requests by hour unauthenticated.
    This can be done with:

    - [
        Personal access token
        ](https://docs.github.com/en/rest/guides/getting-started-with-the-rest-api#using-personal-access-tokens):
        The PAT and username need to be set in the environment variable
        `GH_API_BASIC_AUTH={username}:{personal_access_token}`
    - [
        GitHUB Token
        ](https://docs.github.com/en/actions/security-guides/automatic-token-authentication#about-the-github_token-secret):
        Used in GitHUB Actions, needs to be set in a `GH_TOKEN` environment variable.

    Model states are organised in releases with the following convention:

    - Release name and tags are constructed as `{release_name_prefix}.{state_type.backend}.{state_type.architecture}`
    - Asset names within a release are constructed as
        `{state_type.extra1}.{state_type.extra2}.{training_id}.state.gzip`

    Attributes:
        repository_owner (str): The owner of the GitHUB repository
        repository_name (str): The name of the repository to use as a store
        branch_name (str, optional): The branch used to create new releases holding model state.
            By default we recommend using an
            [orphan branch](https://stackoverflow.com/questions/13202705/when-should-git-orphaned-branches-be-used)
            named `model-store`.
        release_name_prefix (str, optional): The prefix to identify releases containing model weights.
            Defaults to `state`, should not contain a dot.
    """

    repository_owner: str
    repository_name: str
    branch_name: str = "model-store"
    release_name_prefix: str = "state"

    def __post_init__(self):
        if "." in self.release_name_prefix:
            raise ValueError("`release_name_prefix` cannot contain a dot character")

    def gh_release_by_tag_url(self, tag: str) -> str:
        return f"https://api.github.com/repos/{self.repository_owner}/{self.repository_name}/releases/tags/{tag}"

    def get_state_keys(self, state_type: StateType) -> List[StateKey]:
        """List state keys available for a given state type"""
        # Getting release details
        response = call_github_api(
            "get",
            self.gh_release_by_tag_url(
                state_type_to_gh_tag(self.release_name_prefix, state_type)
            ),
        )

        # If not found returns an empty list
        if response.status_code == 404:
            return []

        # Raise if anything else happened
        response.raise_for_status()

        # Getting state keys from payload
        state_keys = [
            gh_asset_name_to_state_key(state_type, asset["name"])
            for asset in response.json()["assets"]
        ]

        # Filtering out invalid assets
        return [sk for sk in state_keys if sk]

    def save(self, model: _ModelType, training_id: str) -> StateKey:
        raise NotImplementedError()

    def load(self, model: _ModelType, state_key: StateKey) -> None:
        return super().load(model, state_key)
