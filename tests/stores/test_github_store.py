import gzip
from collections import OrderedDict
from typing import Tuple
from unittest import mock

import pytest
import requests_mock
from requests.auth import _basic_auth_str

from mlmodule.models.types import ModelWithState
from mlmodule.states import StateKey, StateType
from mlmodule.v2.stores.github import (
    GitHUBReleaseStore,
    call_github_with_auth,
    get_github_basic_auth,
    get_github_token,
    gh_asset_name_to_state_key,
    paginate_github_api,
)


@pytest.fixture(scope="session")
def basic_github_auth():
    username = "test"
    password = "test"

    with mock.patch("mlmodule.v2.stores.github.get_github_basic_auth") as basic:
        with mock.patch("mlmodule.v2.stores.github.get_github_token") as token:
            basic.return_value = (username, password)
            token.return_value = None
            yield _basic_auth_str(username, password)


@pytest.fixture(scope="session")
def token_github_auth():
    gh_token = "abc_aaaaaaaaaa"

    with mock.patch("mlmodule.v2.stores.github.get_github_basic_auth") as basic:
        with mock.patch("mlmodule.v2.stores.github.get_github_token") as token:
            basic.return_value = None
            token.return_value = gh_token
            yield f"Bearer {gh_token}"


@pytest.fixture
def gh_repo_details():
    return "AAA", "test"


@pytest.fixture
def gh_base_url(gh_repo_details: Tuple[str, str]):
    repo_owner, repo_name = gh_repo_details
    return f"https://api.github.com/repos/{repo_owner}/{repo_name}"


@pytest.fixture
def gh_store(gh_repo_details):
    repo_owner, repo_name = gh_repo_details
    return GitHUBReleaseStore(repository_owner=repo_owner, repository_name=repo_name)


def test_github_basic_auth(monkeypatch):
    monkeypatch.setenv("GH_API_BASIC_AUTH", "test:password")
    assert get_github_basic_auth() == ("test", "password")


def test_github_basic_auth_not_set(monkeypatch):
    monkeypatch.delenv("GH_API_BASIC_AUTH", raising=False)
    assert get_github_basic_auth() is None


def test_github_token(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "aaaaaa")
    assert get_github_token() == "aaaaaa"


def test_github_token_not_set(monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert get_github_token() is None


def test_authenticate_github_basic(basic_github_auth: str):
    url = "https://api.github.com/test"
    with requests_mock.Mocker() as m:
        m.get(url, request_headers={"Authorization": basic_github_auth})
        call_github_with_auth("get", url)


def test_authenticate_github_token(token_github_auth: str):
    url = "https://api.github.com/test"
    with requests_mock.Mocker() as m:
        m.get(url, request_headers={"Authorization": token_github_auth})
        call_github_with_auth("get", url)


def test_paginate_github_api():
    url = "https://api.github.com/test"
    with requests_mock.Mocker() as m:
        m.get(
            url,
            headers={"Link": f'{url}?page=2; rel="next"'},
            json=[{"a": 1}, {"b": 2}],
        )
        m.get(
            f"{url}?page=2",
            json=[{"c": 3}],
        )

        assert list(paginate_github_api("get", url)) == [
            {"a": 1},
            {"b": 2},
            {"c": 3},
        ]


def test_paginate_github_api_querystrings():
    url = "https://api.github.com/test"
    params = OrderedDict([("page", 2), ("n_per_page", 100), ("other", 1)])
    querystring = "&".join(f"{key}={value}" for key, value in params.items())
    with requests_mock.Mocker() as m:
        m.get(
            f"{url}?{querystring}",
            headers={"Link": f'{url}?page=3&n_per_page=100; rel="next"'},
            json=[{"a": 1}, {"b": 2}],
        )
        m.get(
            f"{url}?page=3&n_per_page=100",
            json=[{"c": 3}],
        )

        assert list(paginate_github_api("get", url, params=params)) == [
            {"a": 1},
            {"b": 2},
            {"c": 3},
        ]


@pytest.mark.parametrize(
    ("asset_name", "expected_key"),
    [
        (
            "train1.state.gzip",
            StateKey(StateType("pytorch", "resnet18", extra=tuple()), "train1"),
        ),
        (
            "extra1.extra2.train1.state.gzip",
            StateKey(
                StateType("pytorch", "resnet18", extra=("extra1", "extra2")), "train1"
            ),
        ),
        (
            "train1.gzip",
            None,
        ),
    ],
    ids=["no-extra", "extra", "invalid"],
)
def test_gh_asset_name_to_state_key(asset_name: str, expected_key: StateKey):
    assert (
        gh_asset_name_to_state_key(
            StateType(backend="pytorch", architecture="resnet18", extra=("imagenet",)),
            asset_name,
        )
        == expected_key
    )


def test_gh_store_get_state_keys(gh_store: GitHUBReleaseStore, gh_base_url: str):
    state_type = StateType(
        backend="pytorch", architecture="resnet18", extra=("imagenet",)
    )
    url = f"{gh_base_url}/releases/tags/state.pytorch.resnet18"

    with requests_mock.Mocker() as m:
        m.get(
            url,
            json={
                "assets": [
                    {"name": "hello.train.state.gzip"},
                    {"name": "train2.state.gzip"},
                    {"name": "source.zip"},
                ]
            },
        )
        assert set(gh_store.get_state_keys(state_type)) == {
            StateKey(
                state_type=StateType("pytorch", "resnet18", extra=("hello",)),
                training_id="train",
            ),
            StateKey(
                state_type=StateType("pytorch", "resnet18"),
                training_id="train2",
            ),
        }


def test_gh_store_download_state_key(
    gh_store: GitHUBReleaseStore, gh_repo_details: Tuple[str, str]
):
    repo_owner, repo_name = gh_repo_details
    state_key = StateKey(
        state_type=StateType(
            backend="pytorch", architecture="resnet18", extra=("imagenet",)
        ),
        training_id="train1",
    )
    url = (
        f"https://github.com/{repo_owner}/{repo_name}"
        "/releases/download/state.pytorch.resnet18/imagenet.train1.state.gzip"
    )

    # The file exists
    with requests_mock.Mocker() as m:
        m.get(url, content=gzip.compress(b"aaa"))
        assert gh_store.gh_download_state_key(state_key) == b"aaa"

    # The file doesn't exists
    with requests_mock.Mocker() as m:
        m.get(url, status_code=404)
        assert gh_store.gh_download_state_key(state_key) is None


def test_gh_store_load(
    model_with_state: ModelWithState,
    gh_store: GitHUBReleaseStore,
    gh_repo_details: Tuple[str, str],
):
    repo_owner, repo_name = gh_repo_details
    state_key = StateKey(
        state_type=StateType(
            backend="pytorch", architecture="resnet18", extra=("imagenet",)
        ),
        training_id="train1",
    )
    url = (
        f"https://github.com/{repo_owner}/{repo_name}"
        "/releases/download/state.pytorch.resnet18/imagenet.train1.state.gzip"
    )

    # Update the state type to match the one of the URL
    model_with_state._state_type = state_key.state_type  # type: ignore

    # Initial state of the model
    assert model_with_state.get_state() != b"aaa"

    with requests_mock.Mocker() as m:
        m.get(url, content=gzip.compress(b"aaa"))
        gh_store.load(model_with_state, state_key)

    # Makes sure the model state has changed
    assert model_with_state.get_state() == b"aaa"


@pytest.mark.usefixtures("token_github_auth")
def test_get_or_create_release_exists(gh_store: GitHUBReleaseStore, gh_base_url: str):
    state_type = StateType(
        backend="pytorch", architecture="resnet18", extra=("imagenet",)
    )
    release_url = f"{gh_base_url}/releases/tags/state.pytorch.resnet18"

    with requests_mock.Mocker() as m:
        m.get(release_url, json={"id": 1})
        assert gh_store.gh_get_or_create_state_type_release(state_type) == 1


@pytest.mark.usefixtures("token_github_auth")
def test_get_or_create_release_not_exists(
    gh_store: GitHUBReleaseStore, gh_base_url: str
):
    state_type = StateType(
        backend="pytorch", architecture="resnet18", extra=("imagenet",)
    )
    release_url = f"{gh_base_url}/releases/tags/state.pytorch.resnet18"
    post_release_url = f"{gh_base_url}/releases"

    with requests_mock.Mocker() as m:
        m.get(release_url, status_code=404)
        m.post(post_release_url, json={"id": 10})
        assert gh_store.gh_get_or_create_state_type_release(state_type) == 10


@pytest.mark.usefixtures("token_github_auth")
def test_delete_release_asset_exists(gh_store: GitHUBReleaseStore, gh_base_url: str):
    release_id = 10
    asset_name = "test.zip"
    asset_id = 101
    list_release_assets_url = f"{gh_base_url}/releases/{release_id}/assets"
    delete_asset_url = f"{gh_base_url}/releases/assets/{asset_id}"

    with requests_mock.Mocker() as m:
        m.get(
            list_release_assets_url,
            json=[
                {"name": "test2.zip", "id": 111},
                {"name": asset_name, "id": asset_id},
            ],
        )
        m.delete(delete_asset_url, status_code=204)

        assert gh_store.gh_delete_release_asset_if_exists(release_id, asset_name)


@pytest.mark.usefixtures("token_github_auth")
def test_delete_release_asset_not_exists(
    gh_store: GitHUBReleaseStore, gh_base_url: str
):
    release_id = 10
    asset_name = "test.zip"
    asset_id = 101
    list_release_assets_url = f"{gh_base_url}/releases/{release_id}/assets"
    delete_asset_url = f"{gh_base_url}/releases/assets/{asset_id}"

    with requests_mock.Mocker() as m:
        m.get(
            list_release_assets_url,
            json=[{"name": "test2.zip", "id": 111}, {"name": "test1.zip", "id": 121}],
        )
        m.delete(delete_asset_url, status_code=204)

        assert not gh_store.gh_delete_release_asset_if_exists(release_id, asset_name)


@pytest.mark.usefixtures("token_github_auth")
def test_update_state_key_file(
    gh_store: GitHUBReleaseStore,
    model_with_state: ModelWithState,
    gh_repo_details: Tuple[str, str],
):
    release_id = 10
    asset_name = "imagenet.train1.state.gzip"
    repo_owner, repo_name = gh_repo_details
    state_key = StateKey(
        state_type=StateType(
            backend="pytorch", architecture="resnet18", extra=("imagenet",)
        ),
        training_id="train1",
    )
    post_asset_url = f"https://uploads.github.com/repos/{repo_owner}/{repo_name}/releases/{release_id}/assets"
    model_state = model_with_state.get_state()

    with requests_mock.Mocker() as m:
        with mock.patch.object(
            gh_store, "gh_delete_release_asset_if_exists", return_value=False
        ) as s:
            # Mocking the post asset
            m.post(
                f"{post_asset_url}?name={asset_name}&label={asset_name}",
                content=gzip.compress(model_state),
                headers={"Content-Type": "application/gzip"},
            )

            gh_store.gh_update_state_key_file(release_id, state_key, model_state)
            s.assert_called_once_with(release_id, asset_name)
