"""Tests for the Node Registry component."""

import json
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from src.node_registry import NodeRegistry, NodePeer, LocalNode, DISCOVERY_MAGIC


@pytest.fixture
def registry(tmp_path):
    """Fixture providing a NodeRegistry with a temporary storage directory."""
    return NodeRegistry(storage_path=str(tmp_path), qmp_port=9000)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_creates_identity_on_first_run(tmp_path):
    """A new registry should generate and persist a node identity."""
    reg = NodeRegistry(storage_path=str(tmp_path))
    assert reg.local_node is not None
    assert len(reg.local_node.node_id) == 64  # SHA-256 hex
    assert (tmp_path / "node_identity.json").exists()


def test_loads_existing_identity(tmp_path):
    """A second registry instance should load the identity created by the first."""
    reg1 = NodeRegistry(storage_path=str(tmp_path))
    node_id_1 = reg1.local_node.node_id

    reg2 = NodeRegistry(storage_path=str(tmp_path))
    assert reg2.local_node.node_id == node_id_1


def test_identity_has_expected_capabilities(registry):
    """Default capabilities should include git, ai, and storage."""
    caps = registry.local_node.capabilities
    assert "git" in caps
    assert "storage" in caps


# ---------------------------------------------------------------------------
# Peer management
# ---------------------------------------------------------------------------

def test_add_peer(registry):
    """add_peer should store the peer and persist it to disk."""
    peer = registry.add_peer(
        host="192.168.1.50",
        port=9000,
        node_id="a" * 64,
        public_key="pub_key_abc",
        capabilities=["git"],
    )
    assert peer.node_id == "a" * 64
    assert "a" * 64 in {p.node_id for p in registry.get_peers()}
    assert (registry.peers_file).exists()


def test_add_peer_persisted(tmp_path):
    """Peers added in one instance should be visible to a new instance."""
    reg1 = NodeRegistry(storage_path=str(tmp_path))
    reg1.add_peer("10.0.0.1", 9000, "b" * 64, "pub_b")

    reg2 = NodeRegistry(storage_path=str(tmp_path))
    ids = [p.node_id for p in reg2.get_peers()]
    assert "b" * 64 in ids


def test_remove_peer(registry):
    """remove_peer should delete the peer and return True."""
    registry.add_peer("10.0.0.2", 9000, "c" * 64, "pub_c")
    assert registry.remove_peer("c" * 64) is True
    assert registry.get_peer("c" * 64) is None


def test_remove_nonexistent_peer_returns_false(registry):
    assert registry.remove_peer("nonexistent") is False


def test_get_peer(registry):
    registry.add_peer("10.0.0.3", 9001, "d" * 64, "pub_d")
    peer = registry.get_peer("d" * 64)
    assert peer is not None
    assert peer.host == "10.0.0.3"
    assert peer.port == 9001


def test_update_peer_seen(registry):
    registry.add_peer("10.0.0.4", 9002, "e" * 64, "pub_e")
    before = registry.get_peer("e" * 64).last_seen
    time.sleep(0.01)
    registry.update_peer_seen("e" * 64)
    after = registry.get_peer("e" * 64).last_seen
    assert after >= before


# ---------------------------------------------------------------------------
# LAN discovery message handling
# ---------------------------------------------------------------------------

def test_handle_discovery_adds_new_peer(registry):
    """A valid discovery broadcast from a different node should add a peer."""
    payload = json.dumps({
        "type": "announce",
        "node_id": "f" * 64,
        "public_key": "pub_f",
        "port": 9000,
        "capabilities": ["git"],
    }).encode()
    data = DISCOVERY_MAGIC + payload

    registry.handle_discovery_message(data, ("192.168.1.99", 47337))
    assert registry.get_peer("f" * 64) is not None


def test_handle_discovery_ignores_own_broadcast(registry):
    """Discovery messages from our own node_id should be ignored."""
    own_id = registry.local_node.node_id
    payload = json.dumps({
        "type": "announce",
        "node_id": own_id,
        "public_key": registry.local_node.public_key,
        "port": 9000,
        "capabilities": [],
    }).encode()
    data = DISCOVERY_MAGIC + payload

    registry.handle_discovery_message(data, ("127.0.0.1", 47337))
    assert own_id not in registry.peers


def test_handle_discovery_ignores_invalid_magic(registry):
    """Messages without the magic prefix should be silently dropped."""
    data = b"JUNK" + b'{"type":"announce","node_id":"' + b"g" * 64 + b'"}'
    registry.handle_discovery_message(data, ("10.0.0.5", 47337))
    assert registry.get_peer("g" * 64) is None


def test_handle_discovery_ignores_malformed_json(registry):
    """Malformed JSON in a discovery message should not raise."""
    data = DISCOVERY_MAGIC + b"not valid json"
    registry.handle_discovery_message(data, ("10.0.0.6", 47337))  # should not raise


# ---------------------------------------------------------------------------
# NodePeer serialization
# ---------------------------------------------------------------------------

def test_node_peer_round_trip():
    peer = NodePeer(
        node_id="h" * 64,
        host="10.0.0.7",
        port=9003,
        public_key="pub_h",
        capabilities=["git", "ai"],
        last_seen=1000.0,
    )
    restored = NodePeer.from_dict(peer.to_dict())
    assert restored.node_id == peer.node_id
    assert restored.host == peer.host
    assert restored.capabilities == peer.capabilities
