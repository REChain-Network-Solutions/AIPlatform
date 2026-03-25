"""
Titan-BOS Node

The main entry point that boots a federated node by wiring together:
  - NodeRegistry  (identity + peer discovery)
  - QMPService    (TCP mesh messaging)
  - GitFederation (repo management + peer sync)
  - DIDN          (decentralized identity + data store)

Usage:
    import asyncio
    from src.node import Node, NodeConfig

    async def main():
        node = Node()
        await node.start()
        print(node.status())
        await asyncio.sleep(3600)   # keep running
        await node.stop()

    asyncio.run(main())
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.node_registry import NodeRegistry, NodePeer
from src.git_federation import GitFederation, RepoManifest
from src.didn import DIDN
from src.qmp import QMPService, QMPMessage
from src.crypto import verify as _verify_sig

logger = logging.getLogger(__name__)


@dataclass
class NodeConfig:
    """Configuration for a Titan-BOS node."""
    storage_path: str = "~/.titan-bos"
    qmp_port: int = 47338
    git_port: int = 9418
    enable_lan_discovery: bool = True
    enable_git_daemon: bool = True


class Node:
    """
    A Titan-BOS federated node.

    Boots all subsystems and wires them together so that:
    - Peers discovered on LAN automatically connect via QMP
    - On connect, nodes exchange repo lists and DIDN state
    - Git repos stay in sync across peers
    - All data is local-first and persists to storage_path
    """

    def __init__(self, config: Optional[NodeConfig] = None):
        self.config = config or NodeConfig()
        self._storage = Path(self.config.storage_path).expanduser()

        self.registry = NodeRegistry(
            storage_path=str(self._storage),
            qmp_port=self.config.qmp_port,
        )
        self.didn = DIDN(storage_path=str(self._storage / "didn"))
        self.federation = GitFederation(
            repos_path=str(self._storage / "repos"),
            node_id=self.node_id,
            git_port=self.config.git_port,
        )
        self.qmp = QMPService(node_id=self.node_id)

        # node_id -> writer, tracked for targeted sends
        self._peer_writers: Dict[str, asyncio.StreamWriter] = {}
        self._running = False

    # -------------------------------------------------------------------------
    # Identity shortcut
    # -------------------------------------------------------------------------

    @property
    def node_id(self) -> str:
        return self.registry.local_node.node_id

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self):
        """Boot the node: start QMP, git daemon, LAN discovery, connect to peers."""
        self._register_handlers()

        # Start QMP server
        host, port = await self.qmp.start(port=self.config.qmp_port)
        self.registry.local_node.port = port
        self.registry._save_identity()
        logger.info(f"Node {self.node_id[:16]}... QMP listening on {host}:{port}")

        # Start git daemon
        if self.config.enable_git_daemon:
            self.federation.start_server()

        # Start LAN discovery
        if self.config.enable_lan_discovery:
            await self.registry.start_discovery()

        # Connect to already-known peers
        self._running = True
        await self._connect_to_known_peers()

        logger.info(
            f"Node started — peers: {len(self.registry.get_peers())}, "
            f"repos: {len(self.federation.list_repos())}"
        )

    async def stop(self):
        """Gracefully shut down all subsystems."""
        self._running = False
        if self.config.enable_lan_discovery:
            await self.registry.stop_discovery()
        if self.config.enable_git_daemon:
            self.federation.stop_server()
        await self.qmp.stop()
        self._peer_writers.clear()
        logger.info("Node stopped")

    # -------------------------------------------------------------------------
    # QMP message handlers
    # -------------------------------------------------------------------------

    def _register_handlers(self):
        """Register all QMP message type handlers."""
        self.qmp.register_handler("node.handshake", self._handle_handshake)
        self.qmp.register_handler("git.repos.announce", self._handle_git_announce)
        self.qmp.register_handler("didn.sync.request", self._handle_didn_sync_request)
        self.qmp.register_handler("didn.sync.response", self._handle_didn_sync_response)

    async def _handle_handshake(self, message: QMPMessage, writer):
        """
        Peer connected and introduced itself.

        Verifies:
        1. node_id == SHA-256(public_key_bytes) — ID is derived from the key
        2. signature is a valid Ed25519 sig of node_id — proves key ownership

        On success, registers the peer then shares repos and DIDN state.
        """
        info = message.content
        peer_node_id = info.get("node_id")
        if not peer_node_id or peer_node_id == self.node_id:
            return

        pub_key = info.get("public_key", "")
        signature = info.get("signature", "")
        host = writer.get_extra_info("peername")[0]
        port = info.get("qmp_port", 0)
        caps = info.get("capabilities", [])

        # Verify node_id is derived from public_key
        try:
            import hashlib
            expected_id = hashlib.sha256(bytes.fromhex(pub_key)).hexdigest()
        except ValueError:
            logger.warning(f"Rejecting handshake from {host}: public_key is not valid hex")
            return
        if expected_id != peer_node_id:
            logger.warning(f"Rejecting handshake from {host}: node_id/public_key mismatch")
            return

        # Verify the peer holds the matching private key
        if not _verify_sig(pub_key, peer_node_id.encode(), signature):
            logger.warning(f"Rejecting handshake from {host}: signature verification failed")
            return

        self.registry.add_peer(host, port, peer_node_id, pub_key, caps)
        self._peer_writers[peer_node_id] = writer
        logger.info(f"Handshake from {peer_node_id[:16]}... at {host}:{port}")

        # Share our repo list
        await self.qmp.send(
            writer,
            self.qmp.create_message(self.federation.get_announce_payload(), "git.repos.announce"),
        )
        # Ask for their DIDN state
        await self.qmp.send(
            writer,
            self.qmp.create_message({}, "didn.sync.request"),
        )

    async def _handle_git_announce(self, message: QMPMessage, writer):
        """Peer told us what repos they have."""
        host = writer.get_extra_info("peername")[0]
        self.federation.handle_peer_announcement(message.content, host)

    async def _handle_didn_sync_request(self, message: QMPMessage, writer):
        """Peer asked for our DIDN state — send it."""
        state = self.didn.export_state()
        await self.qmp.send(
            writer,
            self.qmp.create_message(state, "didn.sync.response"),
        )

    async def _handle_didn_sync_response(self, message: QMPMessage, writer):
        """Received DIDN state from a peer — merge it (local-first, no overwrite)."""
        content = message.content
        self.didn.merge_from_peer(
            content.get("identities", {}),
            content.get("data", {}),
        )

    # -------------------------------------------------------------------------
    # Outgoing peer connections
    # -------------------------------------------------------------------------

    async def _connect_to_known_peers(self):
        """Spawn connection tasks for all peers in the registry that have a port."""
        for peer in self.registry.get_peers():
            if peer.port > 0:
                asyncio.create_task(self._connect_to_peer(peer))

    async def _connect_to_peer(self, peer: NodePeer):
        """Open a QMP connection to a peer and send our handshake."""
        if peer.node_id in self._peer_writers:
            return  # already connected

        try:
            reader, writer = await asyncio.open_connection(peer.host, peer.port)
            self.qmp.connections.add(writer)
            self._peer_writers[peer.node_id] = writer
            logger.info(f"Connected to peer {peer.node_id[:16]}... at {peer.host}:{peer.port}")

            # Send handshake (includes signature proving we hold the private key)
            await self.qmp.send(
                writer,
                self.qmp.create_message(
                    {
                        "node_id": self.node_id,
                        "public_key": self.registry.local_node.public_key,
                        "qmp_port": self.registry.local_node.port,
                        "capabilities": self.registry.local_node.capabilities,
                        "signature": self.registry.keypair.sign(self.node_id.encode()),
                    },
                    "node.handshake",
                ),
            )

            # Keep reading messages from this peer using the QMP read loop
            await self.qmp._handle_connection(reader, writer)

        except (ConnectionRefusedError, OSError) as e:
            logger.debug(
                f"Could not connect to peer {peer.node_id[:16]}... "
                f"at {peer.host}:{peer.port}: {e}"
            )
            self._peer_writers.pop(peer.node_id, None)

    async def connect_to_peer(self, host: str, port: int):
        """
        Manually connect to a peer by address.

        Use this for WAN peers or when LAN discovery is disabled.
        After connecting, the peer will be registered automatically via handshake.
        """
        temp_peer = NodePeer(
            node_id=f"unknown_{host}_{port}",
            host=host,
            port=port,
            public_key="",
        )
        await self._connect_to_peer(temp_peer)

    # -------------------------------------------------------------------------
    # Convenience API
    # -------------------------------------------------------------------------

    def create_repo(self, name: str, description: str = "") -> Optional[RepoManifest]:
        """Create a new bare git repo on this node."""
        return self.federation.init_repo(name, description)

    def sync_repos(self) -> Dict[str, bool]:
        """Fetch latest changes from all peers for all repos."""
        return self.federation.sync_all()

    def status(self) -> Dict:
        """Return a summary of node status."""
        return {
            "node_id": self.node_id,
            "node_id_short": self.node_id[:16] + "...",
            "qmp_port": self.registry.local_node.port,
            "git_port": self.config.git_port,
            "peers_known": len(self.registry.get_peers()),
            "peers_connected": len(self._peer_writers),
            "repos": len(self.federation.list_repos()),
            "identities": len(self.didn.identities),
            "git_daemon_running": self.federation.server_running(),
        }
