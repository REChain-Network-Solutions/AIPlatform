"""Tests for DIDN persistence and peer sync (extends test_didn.py)."""

import pytest
from pathlib import Path
from src.didn import DIDN, Identity
from src.crypto import KeyPair


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
    """merge_from_peer should add identities that pass Ed25519 verification."""
    kp = KeyPair.generate()
    sig = kp.sign(kp.public_key_hex.encode())

    peer_identities = {
        kp.node_id: {
            "public_key": kp.public_key_hex,
            "signature": sig,
            "timestamp": "2025-01-01T00:00:00",
            "metadata": {},
        }
    }
    persistent_didn.merge_from_peer(peer_identities, {})
    identity = persistent_didn.resolve_identity(kp.node_id)
    assert identity is not None
    assert identity.public_key == kp.public_key_hex


def test_merge_from_peer_rejects_invalid_signature(persistent_didn):
    """Identities with bad signatures should be silently dropped."""
    kp = KeyPair.generate()
    peer_identities = {
        kp.node_id: {
            "public_key": kp.public_key_hex,
            "signature": "deadbeef" * 16,   # wrong signature
            "timestamp": "2025-01-01T00:00:00",
            "metadata": {},
        }
    }
    persistent_didn.merge_from_peer(peer_identities, {})
    assert persistent_didn.resolve_identity(kp.node_id) is None


def test_merge_from_peer_rejects_id_mismatch(persistent_didn):
    """identity_id must equal SHA-256(public_key_bytes); mismatches are dropped."""
    kp = KeyPair.generate()
    sig = kp.sign(kp.public_key_hex.encode())
    wrong_id = "0" * 64   # doesn't match SHA-256(kp.public_key_hex)

    peer_identities = {
        wrong_id: {
            "public_key": kp.public_key_hex,
            "signature": sig,
            "timestamp": "2025-01-01T00:00:00",
            "metadata": {},
        }
    }
    persistent_didn.merge_from_peer(peer_identities, {})
    assert persistent_didn.resolve_identity(wrong_id) is None


def test_merge_from_peer_does_not_overwrite_local(persistent_didn):
    """Local-first: existing entries should not be overwritten by peer data."""
    kp = KeyPair.generate()
    sig = kp.sign(kp.public_key_hex.encode())
    iid = persistent_didn.register_identity(kp.public_key_hex, sig, {"trust": "local"})

    # Build a second valid keypair attempting to register under the same iid (impossible
    # because iid is derived from the public key, but test the local-first guard anyway)
    peer_identities = {
        iid: {
            "public_key": kp.public_key_hex,
            "signature": sig,
            "timestamp": "2099-01-01T00:00:00",
            "metadata": {"trust": "peer"},
        }
    }
    persistent_didn.merge_from_peer(peer_identities, {})

    identity = persistent_didn.resolve_identity(iid)
    assert identity.metadata["trust"] == "local"  # unchanged


def test_export_state_includes_all_data(persistent_didn):
    """export_state should return a snapshot usable by merge_from_peer."""
    kp = KeyPair.generate()
    sig = kp.sign(kp.public_key_hex.encode())
    iid = persistent_didn.register_identity(kp.public_key_hex, sig)
    persistent_didn.store_data(iid, {"key": "value"}, "sig")

    state = persistent_didn.export_state()
    assert iid in state["identities"]
    assert len(state["data"]) == 1
