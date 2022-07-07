import dataclasses
import gzip
import os
import textwrap
from typing import Any, Dict, Generator, List, Optional, Tuple, TypeVar

import requests
from requests.auth import HTTPBasicAuth

from mozuma.models.types import ModelWithState
from mozuma.states import StateKey, StateType
from mozuma.stores.abstract import AbstractStateStore

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


def call_github_with_auth(
    method: str, url: str, force_auth: bool = False, **kwargs
) -> requests.Response:
    """Authenticated call to GitHUB API

    Authentication can be configured with the `GH_TOKEN` or `GH_BASIC_AUTH` environment variables.

    Args:
        method (str): The HTTP method
        url (str): Full URL to call
        force_auth (bool, optional): Whether to force authentication,
            will raise an error if no GitHUB auth is configured.
            Defaults to `False`.

    Raises:
        ValueError: If `force_auth` is True and GitHUB authentication is not configured.
    """
    # Add defaults
    kwargs.setdefault("headers", {})

    # Resolving authentication method
    basic_auth = get_github_basic_auth()
    gh_token = get_github_token()
    if basic_auth:
        kwargs.setdefault("auth", HTTPBasicAuth(*basic_auth))
    elif gh_token:
        kwargs["headers"].setdefault("Authorization", f"Bearer {gh_token}")
    elif force_auth:
        # No auth configured by force_auth is True
        raise ValueError("Cannot resolve GitHUB auth configuration.")

    # Setting recommended headers
    kwargs["headers"]["Accept"] = "application/vnd.github.v3+json"

    return getattr(requests, method)(url, **kwargs)


def paginate_github_api(
    method: str, url: str, **kwargs
) -> Generator[_JsonType, None, None]:
    while True:
        # Calling GitHUB
        response = call_github_with_auth(method, url, **kwargs)
        response.raise_for_status()

        # Returning results
        for r in response.json():
            yield r

        # Stopping if no next url
        if "next" not in response.links:
            break

        # Getting next URL
        url = response.links["next"]["url"]

        # Removing querystring if exists
        if "params" in kwargs:
            del kwargs["params"]


def state_type_to_gh_tag(release_name_prefix: str, state_type: StateType) -> str:
    """Formats a state type to a GitHUB release tag"""
    return f"{release_name_prefix}.{state_type.backend}.{state_type.architecture}"


def state_key_to_gh_asset_name(state_key: StateKey) -> str:
    """From a state key returns the name of the asset in the corresponding release"""
    extra = state_key.state_type.extra
    filename = ".".join(extra + (state_key.training_id,))
    return f"{filename}.state.gzip"


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


def state_type_to_release_body(state_type: StateType) -> str:
    return textwrap.dedent(
        f"""
    This release contains model state files for model with the following characteristics:

    - **Backend**: `{state_type.backend}`
    - **Architecture**: `{state_type.architecture}`
    """
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

    def gh_releases_url(self) -> str:
        return f"https://api.github.com/repos/{self.repository_owner}/{self.repository_name}/releases"

    def gh_release_by_tag_url(self, tag: str) -> str:
        return f"{self.gh_releases_url()}/tags/{tag}"

    def gh_release_assets_url(self, release_id: int) -> str:
        return f"{self.gh_releases_url()}/{release_id}/assets"

    def gh_download_state_key_url(self, state_key: StateKey) -> str:
        """Download URL for the state key"""
        return (
            f"https://github.com/{self.repository_owner}/{self.repository_name}"
            f"/releases/download/{state_type_to_gh_tag(self.release_name_prefix, state_key.state_type)}"
            f"/{state_key_to_gh_asset_name(state_key)}"
        )

    def gh_download_state_key(self, state_key: StateKey) -> Optional[bytes]:
        """Download the state key from GitHUB

        Returns:
            BinaryIO | None: The binary stream of the weights or None if the weights doesn't exist"""
        response = call_github_with_auth(
            "get", self.gh_download_state_key_url(state_key)
        )

        # When state does not exists
        if response.status_code == 404:
            return None

        # For any other error
        response.raise_for_status()

        return gzip.decompress(response.content)

    def gh_get_release_by_tag(self, tag: str) -> Optional[_JsonType]:
        """Returns the release JSON object from a tag

        Args:
            tag (str): GitHUB tag to search a matching release

        Returns:
            _JsonType | None: The JSON asset payload if found otherwise None.
        """
        response = call_github_with_auth(
            "get",
            self.gh_release_by_tag_url(tag),
        )
        if response.status_code == 404:
            return None

        # Raise if unexpected error occurred
        response.raise_for_status()

        return response.json()

    def get_state_keys(self, state_type: StateType) -> List[StateKey]:
        """List state keys available for a given state type"""
        # Getting release details
        release = self.gh_get_release_by_tag(
            state_type_to_gh_tag(self.release_name_prefix, state_type)
        )

        # If not found returns an empty list
        if release is None:
            return []

        # Getting state keys from payload
        state_keys = [
            gh_asset_name_to_state_key(state_type, asset["name"])
            for asset in release["assets"]
        ]

        # Filtering out invalid assets
        return [sk for sk in state_keys if sk]

    def load(self, model: _ModelType, state_key: StateKey) -> None:
        # Making sure the model and state key are compatible
        super().load(model, state_key)

        # Downloading the state
        binary_state = self.gh_download_state_key(state_key)
        if binary_state is None:
            raise ValueError(
                f"The given state key does not exist in the store: {state_key}"
            )

        # Applying to the model
        model.set_state(binary_state)

    def gh_get_or_create_state_type_release(self, state_type: StateType) -> int:
        """Creates a release to hold model states for a state type

        Args:
            state_type (StateType): The state type that the release will represent

        Returns:
            str: The release ID that can be used to upload assets
        """
        state_type_tag = state_type_to_gh_tag(self.release_name_prefix, state_type)

        # Getting the release for the state type
        release = self.gh_get_release_by_tag(state_type_tag)

        # If exists return the id
        if release is not None:
            return release["id"]

        # Otherwise create the new release
        response = call_github_with_auth(
            "post",
            self.gh_releases_url(),
            json={
                "tag_name": state_type_tag,
                "target_commitish": self.branch_name,
                "name": state_type_tag,
                "body": state_type_to_release_body(state_type),
                "draft": False,
                "prerelease": False,
                "generate_release_notes": False,
            },
            force_auth=True,
        )
        response.raise_for_status()
        return response.json()["id"]

    def gh_get_release_asset_id_by_name(
        self, release_id: int, asset_name: str
    ) -> Optional[int]:
        """Finds the asset id in a release from the asset name

        Args:
            release_id (int): The identifier of the release
            asset_name (str): The name of the asset to find

        Returns:
            int | None: The asset id if it exists
        """
        release_assets = paginate_github_api(
            "get", self.gh_release_assets_url(release_id)
        )
        # Return the first asset matching
        return next(
            (asset["id"] for asset in release_assets if asset["name"] == asset_name),
            None,
        )

    def gh_delete_release_asset_if_exists(
        self, release_id: int, asset_name: str
    ) -> bool:
        """Deletes a release asset by name if exists

        Args:
            release_id (int): The identifier of the release
            asset_name (str): The name of the asset to delete

        Returns:
            bool: True if the asset needed to be deleted
        """
        asset_id = self.gh_get_release_asset_id_by_name(release_id, asset_name)
        if asset_id is None:
            return False

        response = call_github_with_auth(
            "delete", f"{self.gh_releases_url()}/assets/{asset_id}", force_auth=True
        )
        response.raise_for_status()
        return True

    def gh_update_state_key_file(
        self, release_id: int, state_key: StateKey, binary_state: bytes
    ) -> None:
        """Update the asset file for a state key in a release

        Args:
            release_id (int): The id of the GitHUB release
            state_key (StateKey): The state key of the model to save
            binary_state (bytes): The model state as binary
        """
        asset_name = state_key_to_gh_asset_name(state_key)
        self.gh_delete_release_asset_if_exists(release_id, asset_name)

        response = call_github_with_auth(
            "post",
            (
                "https://uploads.github.com/repos/"
                f"{self.repository_owner}/{self.repository_name}/releases/{release_id}/assets"
            ),
            force_auth=True,
            data=gzip.compress(binary_state),
            headers={"Content-Type": "application/gzip"},
            params={"name": asset_name, "label": asset_name},
        )
        response.raise_for_status()

    def save(self, model: _ModelType, training_id: str) -> StateKey:
        # Getting state key
        state_key = super().save(model, training_id)

        # Getting release ID for the state type
        release_id = self.gh_get_or_create_state_type_release(state_key.state_type)

        # Updating the release asset file
        self.gh_update_state_key_file(release_id, state_key, model.get_state())

        return state_key
