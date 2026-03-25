"""Tests for DIDN persistence and peer sync (extends test_didn.py)."""

import pytest
from pathlib import Path
from src.didn import DIDN, Identity


@pytest.fixture
def persistent_didn(tmp_path):
    return DIDN(storage_path=str(tmp_path / "didn"))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_identity_survives_restart(tmp_path):
    """Identities registered in one instance should load in the next."""
    d1 = DIDN(storage_path=str(tmp_path / "didn"))
    iid = d1.register_identity("pub_key_1", "sig_1", {"name": "Alice"})

    d2 = DIDN(storage_path=str(tmp_path / "didn"))
    identity = d2.resolve_identity(iid)
    assert identity is not None
    assert identity.public_key == "pub_key_1"
    assert identity.metadata["name"] == "Alice"


def test_data_survives_restart(tmp_path):
    """Stored data should be available after reloading from disk."""
    d1 = DIDN(storage_path=str(tmp_path / "didn"))
    iid = d1.register_identity("pub_key_2", "sig_2")
    did = d1.store_data(iid, {"msg": "hello"}, "data_sig")

    d2 = DIDN(storage_path=str(tmp_path / "didn"))
    record = d2.resolve_data(did)
    assert record is not None
    assert record["data"]["msg"] == "hello"


def test_storage_files_created(tmp_path):
    """Persistence files should be created after writing data."""
    d = DIDN(storage_path=str(tmp_path / "didn"))
    d.register_identity("pub_key_3", "sig_3")
    assert (tmp_path / "didn" / "identities.json").exists()


def test_in_memory_mode_no_files(tmp_path):
    """Without storage_path, no files should be written."""
    d = DIDN()
    d.register_identity("pub_key_4", "sig_4")
    assert not list(tmp_path.glob("**/*.json"))


# ---------------------------------------------------------------------------
# Peer sync
# ---------------------------------------------------------------------------

def test_merge_from_peer_adds_new_identities(persistent_didn):
    """merge_from_peer should add identities not already present."""
    peer_state = {
        "identities": {
            "a" * 64: {
                "public_key": "peer_pub",
                "signature": "peer_sig",
                "timestamp": "2025-01-01T00:00:00",
                "metadata": {},
            }
        },
        "data": {},
    }
    persistent_didn.merge_from_peer(peer_state["identities"], peer_state["data"])
    identity = persistent_didn.resolve_identity("a" * 64)
    assert identity is not None
    assert identity.public_key == "peer_pub"


def test_merge_from_peer_does_not_overwrite_local(persistent_didn):
    """Local-first: existing entries should not be overwritten by peer data."""
    iid = persistent_didn.register_identity("local_pub", "local_sig", {"trust": "local"})

    peer_identities = {
        iid: {
            "public_key": "tampered_pub",
            "signature": "tampered_sig",
            "timestamp": "2099-01-01T00:00:00",
            "metadata": {"trust": "peer"},
        }
    }
    persistent_didn.merge_from_peer(peer_identities, {})

    identity = persistent_didn.resolve_identity(iid)
    assert identity.public_key == "local_pub"  # unchanged


def test_export_state_includes_all_data(persistent_didn):
    """export_state should return a snapshot usable by merge_from_peer."""
    iid = persistent_didn.register_identity("export_pub", "export_sig")
    persistent_didn.store_data(iid, {"key": "value"}, "sig")

    state = persistent_didn.export_state()
    assert iid in state["identities"]
    assert len(state["data"]) == 1
