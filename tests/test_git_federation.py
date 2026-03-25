"""Tests for the Git Federation component."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.git_federation import GitFederation, RepoManifest


NODE_ID = "a" * 64
PEER_NODE_ID = "b" * 64


@pytest.fixture
def fed(tmp_path):
    """Fixture providing a GitFederation instance with a temp repo directory."""
    return GitFederation(
        repos_path=str(tmp_path / "repos"),
        node_id=NODE_ID,
        git_port=9418,
    )


# ---------------------------------------------------------------------------
# Repo init
# ---------------------------------------------------------------------------

def test_init_repo_creates_bare_repo(fed, tmp_path):
    """init_repo should create a bare git repo on disk."""
    manifest = fed.init_repo("myproject", "A test repo")
    assert manifest is not None
    assert manifest.name == "myproject"
    assert manifest.owner_node_id == NODE_ID
    assert (tmp_path / "repos" / "myproject.git").is_dir()


def test_init_repo_persists_manifest(fed, tmp_path):
    """Manifest should be written to _manifests.json."""
    fed.init_repo("alpha")
    assert (tmp_path / "repos" / "_manifests.json").exists()


def test_init_repo_duplicate_returns_existing(fed):
    """Calling init_repo twice for the same name should not fail."""
    m1 = fed.init_repo("beta")
    m2 = fed.init_repo("beta")
    assert m1 is not None
    assert m2 is not None
    assert m1.name == m2.name


def test_list_repos(fed):
    fed.init_repo("repo1")
    fed.init_repo("repo2")
    names = [m.name for m in fed.list_repos()]
    assert "repo1" in names
    assert "repo2" in names


def test_get_repo(fed):
    fed.init_repo("gamma", "desc")
    m = fed.get_repo("gamma")
    assert m is not None
    assert m.description == "desc"


def test_get_repo_nonexistent(fed):
    assert fed.get_repo("doesnotexist") is None


def test_manifests_survive_reload(tmp_path):
    """Manifests written to disk should load correctly in a new instance."""
    fed1 = GitFederation(str(tmp_path / "repos"), NODE_ID)
    fed1.init_repo("delta")

    fed2 = GitFederation(str(tmp_path / "repos"), NODE_ID)
    assert fed2.get_repo("delta") is not None


# ---------------------------------------------------------------------------
# Peer URL helpers
# ---------------------------------------------------------------------------

def test_peer_url(fed):
    url = fed.peer_url("192.168.1.10", "myrepo")
    assert url == "git://192.168.1.10:9418/myrepo.git"


def test_local_url(fed, tmp_path):
    url = fed._local_url("myrepo")
    assert url.endswith("myrepo.git")


# ---------------------------------------------------------------------------
# QMP integration
# ---------------------------------------------------------------------------

def test_get_announce_payload(fed):
    fed.init_repo("epsilon", "my repo")
    payload = fed.get_announce_payload()
    assert payload["node_id"] == NODE_ID
    assert payload["git_port"] == 9418
    repo_names = [r["name"] for r in payload["repos"]]
    assert "epsilon" in repo_names


def test_get_announce_payload_only_owned_repos(fed):
    """Only repos owned by this node should appear in the announce payload."""
    fed.init_repo("owned")
    # Manually insert a repo owned by a peer
    from src.git_federation import RepoManifest
    import time
    fed.manifests["peer_repo"] = RepoManifest(
        name="peer_repo",
        description="",
        owner_node_id=PEER_NODE_ID,
        created_at=time.time(),
    )
    payload = fed.get_announce_payload()
    names = [r["name"] for r in payload["repos"]]
    assert "owned" in names
    assert "peer_repo" not in names


def test_handle_peer_announcement_new_repo(fed):
    """Announcing a repo we don't have should add it to manifests."""
    payload = {
        "node_id": PEER_NODE_ID,
        "git_port": 9418,
        "repos": [{"name": "shared", "description": "shared repo"}],
    }
    fed.handle_peer_announcement(payload, "192.168.1.20")
    m = fed.get_repo("shared")
    assert m is not None
    assert m.owner_node_id == PEER_NODE_ID
    assert PEER_NODE_ID in m.remotes


def test_handle_peer_announcement_known_repo(fed):
    """Announcing a repo we already have should register the peer as a remote."""
    fed.init_repo("zeta")

    payload = {
        "node_id": PEER_NODE_ID,
        "git_port": 9418,
        "repos": [{"name": "zeta", "description": ""}],
    }

    with patch.object(fed, "add_peer_remote", return_value=True) as mock_add:
        fed.handle_peer_announcement(payload, "192.168.1.21")
        mock_add.assert_called_once_with("zeta", PEER_NODE_ID, "192.168.1.21")


def test_handle_peer_announcement_missing_node_id(fed):
    """Announcements without a node_id should be silently ignored."""
    fed.handle_peer_announcement({"repos": [{"name": "x"}]}, "10.0.0.1")
    assert fed.get_repo("x") is None


# ---------------------------------------------------------------------------
# RepoManifest serialization
# ---------------------------------------------------------------------------

def test_repo_manifest_round_trip():
    import time
    m = RepoManifest(
        name="test",
        description="desc",
        owner_node_id=NODE_ID,
        created_at=time.time(),
        remotes={NODE_ID: "/local/path"},
    )
    restored = RepoManifest.from_dict(m.to_dict())
    assert restored.name == m.name
    assert restored.owner_node_id == m.owner_node_id
    assert restored.remotes == m.remotes
