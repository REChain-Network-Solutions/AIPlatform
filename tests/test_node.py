"""Tests for the Node boot and message wiring."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from src.node import Node, NodeConfig
from src.qmp import QMPMessage


NODE_A_ID = "a" * 64
NODE_B_ID = "b" * 64


@pytest.fixture
def config(tmp_path):
    return NodeConfig(
        storage_path=str(tmp_path),
        qmp_port=0,          # OS-assigned port
        git_port=9419,
        enable_lan_discovery=False,  # keep tests offline
        enable_git_daemon=False,     # no real git daemon in unit tests
    )


@pytest.fixture
def node(config):
    return Node(config)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_node_has_identity(node):
    assert len(node.node_id) == 64


def test_node_subsystems_initialized(node):
    assert node.registry is not None
    assert node.federation is not None
    assert node.didn is not None
    assert node.qmp is not None


def test_node_status_shape(node):
    s = node.status()
    assert "node_id" in s
    assert "node_id_short" in s
    assert "peers_known" in s
    assert "peers_connected" in s
    assert "repos" in s
    assert "identities" in s


# ---------------------------------------------------------------------------
# create_repo
# ---------------------------------------------------------------------------

def test_create_repo(node, tmp_path):
    manifest = node.create_repo("test-repo", "A test")
    assert manifest is not None
    assert manifest.name == "test-repo"
    assert node.status()["repos"] == 1


# ---------------------------------------------------------------------------
# Handler registration
# ---------------------------------------------------------------------------

def test_all_handlers_registered(node):
    node._register_handlers()
    handlers = node.qmp.message_handlers
    assert "node.handshake" in handlers
    assert "git.repos.announce" in handlers
    assert "didn.sync.request" in handlers
    assert "didn.sync.response" in handlers


# ---------------------------------------------------------------------------
# _handle_handshake
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handshake_registers_peer(node):
    writer = _mock_writer("192.168.1.10")
    node.qmp.send = AsyncMock()

    msg = QMPMessage(
        content={
            "node_id": NODE_B_ID,
            "public_key": "pub_b",
            "qmp_port": 9000,
            "capabilities": ["git"],
        },
        sender_id=NODE_B_ID,
        message_type="node.handshake",
        timestamp=1000.0,
    )
    await node._handle_handshake(msg, writer)

    peer = node.registry.get_peer(NODE_B_ID)
    assert peer is not None
    assert peer.host == "192.168.1.10"
    assert peer.port == 9000


@pytest.mark.asyncio
async def test_handshake_sends_git_announce_and_didn_request(node):
    writer = _mock_writer("192.168.1.11")
    node.qmp.send = AsyncMock()

    msg = QMPMessage(
        content={
            "node_id": NODE_B_ID,
            "public_key": "pub_b",
            "qmp_port": 9001,
            "capabilities": [],
        },
        sender_id=NODE_B_ID,
        message_type="node.handshake",
        timestamp=1000.0,
    )
    await node._handle_handshake(msg, writer)

    assert node.qmp.send.await_count == 2
    sent_types = [c.args[1].message_type for c in node.qmp.send.await_args_list]
    assert "git.repos.announce" in sent_types
    assert "didn.sync.request" in sent_types


@pytest.mark.asyncio
async def test_handshake_ignores_own_node_id(node):
    """A handshake from our own node_id should be silently ignored."""
    writer = _mock_writer("127.0.0.1")
    node.qmp.send = AsyncMock()

    msg = QMPMessage(
        content={
            "node_id": node.node_id,
            "public_key": "self",
            "qmp_port": 9000,
            "capabilities": [],
        },
        sender_id=node.node_id,
        message_type="node.handshake",
        timestamp=1000.0,
    )
    await node._handle_handshake(msg, writer)

    node.qmp.send.assert_not_awaited()
    assert node.registry.get_peer(node.node_id) is None


# ---------------------------------------------------------------------------
# _handle_git_announce
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_git_announce_calls_federation(node):
    writer = _mock_writer("192.168.1.12")
    payload = {
        "node_id": NODE_B_ID,
        "git_port": 9418,
        "repos": [{"name": "shared", "description": ""}],
    }
    msg = QMPMessage(
        content=payload,
        sender_id=NODE_B_ID,
        message_type="git.repos.announce",
        timestamp=1000.0,
    )

    with patch.object(node.federation, "handle_peer_announcement") as mock_announce:
        await node._handle_git_announce(msg, writer)
        mock_announce.assert_called_once_with(payload, "192.168.1.12")


# ---------------------------------------------------------------------------
# _handle_didn_sync_request / response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_didn_sync_request_sends_state(node):
    writer = _mock_writer("192.168.1.13")
    node.qmp.send = AsyncMock()

    # Seed some data
    iid = node.didn.register_identity("pub_x", "sig_x")

    msg = QMPMessage(
        content={},
        sender_id=NODE_B_ID,
        message_type="didn.sync.request",
        timestamp=1000.0,
    )
    await node._handle_didn_sync_request(msg, writer)

    node.qmp.send.assert_awaited_once()
    sent_msg = node.qmp.send.await_args.args[1]
    assert sent_msg.message_type == "didn.sync.response"
    assert iid in sent_msg.content.get("identities", {})


@pytest.mark.asyncio
async def test_didn_sync_response_merges_state(node):
    peer_state = {
        "identities": {
            "c" * 64: {
                "public_key": "peer_pub",
                "signature": "peer_sig",
                "timestamp": "2025-01-01T00:00:00",
                "metadata": {},
            }
        },
        "data": {},
    }
    writer = _mock_writer("192.168.1.14")
    msg = QMPMessage(
        content=peer_state,
        sender_id=NODE_B_ID,
        message_type="didn.sync.response",
        timestamp=1000.0,
    )
    await node._handle_didn_sync_response(msg, writer)

    identity = node.didn.resolve_identity("c" * 64)
    assert identity is not None
    assert identity.public_key == "peer_pub"


# ---------------------------------------------------------------------------
# Node start / stop (mocked subsystems)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_node_start_stop(config):
    node = Node(config)

    with patch.object(node.qmp, "start", new_callable=AsyncMock, return_value=("0.0.0.0", 12345)):
        with patch.object(node.qmp, "stop", new_callable=AsyncMock):
            await node.start()
            s = node.status()
            assert s["qmp_port"] == 12345
            assert s["git_daemon_running"] is False  # disabled in config
            await node.stop()


@pytest.mark.asyncio
async def test_node_start_registers_handlers(config):
    node = Node(config)

    with patch.object(node.qmp, "start", new_callable=AsyncMock, return_value=("0.0.0.0", 0)):
        with patch.object(node.qmp, "stop", new_callable=AsyncMock):
            await node.start()
            assert "node.handshake" in node.qmp.message_handlers
            await node.stop()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mock_writer(host: str):
    writer = AsyncMock()
    writer.get_extra_info = MagicMock(return_value=(host, 12345))
    writer.drain = AsyncMock()
    return writer
