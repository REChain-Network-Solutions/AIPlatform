"""
Node Registry

Manages local node identity and peer discovery for the federated network.
Supports local-first operation: works fully offline, discovers peers on LAN
via UDP broadcast, and allows manual peer addition for WAN connections.

Each node generates a keypair-derived identity on first run, persisted to
~/.titan-bos/ by default.
"""

import asyncio
import json
import hashlib
import secrets
import socket
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DISCOVERY_PORT = 47337
DISCOVERY_MAGIC = b"TITAN-BOS-NODE"


@dataclass
class NodePeer:
    """Represents a known peer in the network."""
    node_id: str
    host: str
    port: int
    public_key: str
    capabilities: List[str] = field(default_factory=list)
    last_seen: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'NodePeer':
        return cls(**data)


@dataclass
class LocalNode:
    """Represents this node's identity."""
    node_id: str
    public_key: str
    private_key_hint: str
    host: str
    port: int
    capabilities: List[str] = field(default_factory=list)
    created_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'LocalNode':
        return cls(**data)


class NodeRegistry:
    """
    Manages node identity and peer discovery.

    Supports local-first operation: works fully offline, discovers peers
    on LAN via UDP broadcast, and allows manual peer addition for WAN.
    """

    def __init__(self, storage_path: Optional[str] = None, qmp_port: int = 0):
        self.storage_path = Path(storage_path) if storage_path else Path.home() / ".titan-bos"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.peers_file = self.storage_path / "peers.json"
        self.identity_file = self.storage_path / "node_identity.json"
        self.qmp_port = qmp_port

        self.peers: Dict[str, NodePeer] = {}
        self.local_node: Optional[LocalNode] = None
        self._discovery_transport = None

        self._load_peers()
        self._load_or_create_identity()

    def _load_or_create_identity(self):
        """Load existing identity or create a new one."""
        if self.identity_file.exists():
            with open(self.identity_file) as f:
                self.local_node = LocalNode.from_dict(json.load(f))
            logger.info(f"Loaded node identity: {self.local_node.node_id[:16]}...")
        else:
            self._create_identity()

    def _create_identity(self):
        """Generate a new node identity."""
        private_hint = secrets.token_hex(32)
        public_key = hashlib.sha256(private_hint.encode()).hexdigest()
        node_id = hashlib.sha256(f"node:{public_key}".encode()).hexdigest()

        self.local_node = LocalNode(
            node_id=node_id,
            public_key=public_key,
            private_key_hint=private_hint,
            host="0.0.0.0",
            port=self.qmp_port,
            capabilities=["git", "ai", "storage"],
            created_at=time.time(),
        )
        self._save_identity()
        logger.info(f"Created new node identity: {node_id[:16]}...")

    def _save_identity(self):
        """Persist node identity to disk."""
        with open(self.identity_file, "w") as f:
            json.dump(self.local_node.to_dict(), f, indent=2)

    def _load_peers(self):
        """Load known peers from disk."""
        if self.peers_file.exists():
            with open(self.peers_file) as f:
                data = json.load(f)
            self.peers = {k: NodePeer.from_dict(v) for k, v in data.items()}
            logger.info(f"Loaded {len(self.peers)} known peers")

    def _save_peers(self):
        """Persist known peers to disk."""
        with open(self.peers_file, "w") as f:
            json.dump({k: v.to_dict() for k, v in self.peers.items()}, f, indent=2)

    def add_peer(
        self,
        host: str,
        port: int,
        node_id: str,
        public_key: str,
        capabilities: Optional[List[str]] = None,
    ) -> NodePeer:
        """Manually add a peer (for WAN connections or static config)."""
        peer = NodePeer(
            node_id=node_id,
            host=host,
            port=port,
            public_key=public_key,
            capabilities=capabilities or [],
            last_seen=time.time(),
        )
        self.peers[node_id] = peer
        self._save_peers()
        logger.info(f"Added peer {node_id[:16]}... at {host}:{port}")
        return peer

    def remove_peer(self, node_id: str) -> bool:
        """Remove a peer from the registry."""
        if node_id in self.peers:
            del self.peers[node_id]
            self._save_peers()
            return True
        return False

    def get_peers(self) -> List[NodePeer]:
        """Return all known peers."""
        return list(self.peers.values())

    def get_peer(self, node_id: str) -> Optional[NodePeer]:
        """Get a specific peer by node_id."""
        return self.peers.get(node_id)

    def update_peer_seen(self, node_id: str):
        """Update last_seen timestamp for a peer."""
        if node_id in self.peers:
            self.peers[node_id].last_seen = time.time()
            self._save_peers()

    async def start_discovery(self):
        """Start LAN peer discovery via UDP broadcast."""
        loop = asyncio.get_event_loop()

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setblocking(False)

        try:
            sock.bind(("", DISCOVERY_PORT))
        except OSError:
            logger.warning(
                f"Could not bind to discovery port {DISCOVERY_PORT}, "
                "LAN discovery disabled. Use add_peer() for manual connections."
            )
            sock.close()
            return

        transport, _ = await loop.create_datagram_endpoint(
            lambda: _DiscoveryProtocol(self),
            sock=sock,
        )
        self._discovery_transport = transport
        logger.info(f"LAN discovery listening on UDP port {DISCOVERY_PORT}")
        await self._broadcast_announce()

    async def stop_discovery(self):
        """Stop LAN peer discovery."""
        if self._discovery_transport:
            self._discovery_transport.close()
            self._discovery_transport = None

    async def _broadcast_announce(self):
        """Broadcast this node's presence on the LAN."""
        if not self._discovery_transport:
            return

        payload = DISCOVERY_MAGIC + json.dumps({
            "type": "announce",
            "node_id": self.local_node.node_id,
            "public_key": self.local_node.public_key,
            "port": self.local_node.port,
            "capabilities": self.local_node.capabilities,
        }).encode()

        self._discovery_transport.sendto(payload, ("<broadcast>", DISCOVERY_PORT))
        logger.debug("Broadcast node announcement on LAN")

    def handle_discovery_message(self, data: bytes, addr: tuple):
        """Handle an incoming discovery UDP broadcast."""
        if not data.startswith(DISCOVERY_MAGIC):
            return

        try:
            msg = json.loads(data[len(DISCOVERY_MAGIC):])
            if msg.get("type") != "announce":
                return

            node_id = msg["node_id"]
            host = addr[0]

            if node_id == self.local_node.node_id:
                return  # ignore our own broadcasts

            if node_id not in self.peers:
                peer = NodePeer(
                    node_id=node_id,
                    host=host,
                    port=msg["port"],
                    public_key=msg["public_key"],
                    capabilities=msg.get("capabilities", []),
                    last_seen=time.time(),
                )
                self.peers[node_id] = peer
                self._save_peers()
                logger.info(f"Discovered new peer {node_id[:16]}... at {host}:{msg['port']}")
            else:
                self.update_peer_seen(node_id)

        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Invalid discovery message from {addr}: {e}")


class _DiscoveryProtocol(asyncio.DatagramProtocol):
    """UDP datagram handler for LAN node discovery."""

    def __init__(self, registry: NodeRegistry):
        self.registry = registry

    def datagram_received(self, data: bytes, addr: tuple):
        self.registry.handle_discovery_message(data, addr)

    def error_received(self, exc: Exception):
        logger.debug(f"Discovery protocol error: {exc}")
