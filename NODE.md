# Running a Titan-BOS Node

This guide covers setting up and running Titan-BOS as a **local-first federated node** — no central server required.

---

## Concept

Each Titan-BOS node is a peer in a self-organizing network:

```
Node A ──── QMP ────► Node B
  │                     │
  └── git sync ─────────┘
  │                     │
  └── DIDN sync ─────────┘
```

- **Local-first**: all data lives on your node. Nothing requires the internet to work.
- **Federated git**: repos are bare git repos on each node, synced peer-to-peer.
- **Identity**: each node generates a keypair-derived identity on first run (`~/.titan-bos/`).
- **Discovery**: nodes find each other on LAN via UDP broadcast. WAN peers are added manually.

---

## Quick Start

### 1. Install Python dependencies

```bash
pip install -e ".[dev]"
```

### 2. Run the test suite

```bash
python -m pytest tests/ -v
```

### 3. Start a node (Python REPL or script)

```python
import asyncio
from src.node_registry import NodeRegistry
from src.git_federation import GitFederation
from src.qmp import QMPService

# Node identity + peer discovery (persists to ~/.titan-bos/)
registry = NodeRegistry()
print(f"Node ID: {registry.local_node.node_id[:16]}...")

# Git federation
fed = GitFederation(
    repos_path="~/.titan-bos/repos",
    node_id=registry.local_node.node_id,
)

# Create a repo on this node
fed.init_repo("my-business-data", "Internal BOS data")

# Start LAN discovery (finds peers automatically)
async def run():
    await registry.start_discovery()
    await asyncio.sleep(3600)  # keep running

asyncio.run(run())
```

---

## Core Modules

### `src/node_registry`

Manages this node's identity and known peers.

```python
from src.node_registry import NodeRegistry

reg = NodeRegistry()                        # loads or creates ~/.titan-bos/node_identity.json
reg.add_peer("192.168.1.50", 9000, node_id, pub_key)  # WAN / static peer
peers = reg.get_peers()
```

Stored at `~/.titan-bos/`:
- `node_identity.json` — this node's keypair-derived ID
- `peers.json` — known peers

### `src/git_federation`

Manages bare git repos, announced and synced peer-to-peer.

```python
from src.git_federation import GitFederation

fed = GitFederation("~/.titan-bos/repos", node_id=reg.local_node.node_id)

fed.init_repo("invoices", "Invoice data")          # create local bare repo
fed.clone_from_peer("invoices", "192.168.1.50", peer_node_id)  # clone from peer
fed.sync_from_peer("invoices", peer_node_id)       # fetch latest from peer
fed.sync_all()                                     # sync all repos from all peers

# Build a QMP announcement payload to broadcast to peers
payload = fed.get_announce_payload()

# Handle an announcement received from a peer
fed.handle_peer_announcement(payload, peer_host="192.168.1.50")
```

Repos are stored as bare git repos: `~/.titan-bos/repos/myrepo.git`

### `src/didn`

Decentralized identity and data store. Works in-memory or with disk persistence.

```python
from src.didn import DIDN

didn = DIDN(storage_path="~/.titan-bos/didn")  # persistent

iid = didn.register_identity(public_key, signature, metadata={"name": "Alice"})
did = didn.store_data(iid, {"type": "invoice", "amount": 100}, signature)

# Sync with a peer
state = didn.export_state()                    # send this to a peer
didn.merge_from_peer(peer_state["identities"], peer_state["data"])
```

### `src/qmp`

TCP mesh protocol for node-to-node messages.

```python
from src.qmp import QMPService, QMPMessage

svc = QMPService(node_id=reg.local_node.node_id)
svc.register_handler("git.repos.announce", handle_git_announce)

async def run():
    host, port = await svc.start()
    print(f"QMP listening on {host}:{port}")
```

### `src/self_contained_cicd`

Autonomous build/test/deploy pipeline on each node.

```python
from src.self_contained_cicd import SelfContainedCICD

cicd = SelfContainedCICD(project_root=".")
# Configure via .cicd/config.json
await cicd.run_pipeline()
```

---

## Adding a WAN Peer

When nodes are not on the same LAN, add them manually:

```python
registry.add_peer(
    host="peer.example.com",
    port=9000,
    node_id="<64-char-hex-node-id>",
    public_key="<hex-public-key>",
    capabilities=["git", "storage"],
)
```

Exchange `node_id` and `public_key` values out-of-band (signal, paper, QR code).

---

## Privacy Notes

- No data leaves your node unless you explicitly sync with a peer you added.
- LAN discovery (UDP broadcast) only reveals your `node_id`, `public_key`, and `port` — no business data.
- Disable LAN discovery entirely by simply not calling `registry.start_discovery()`.
- Repos are standard bare git repos — encrypt with `git-crypt` or similar if needed.

---

## Directory Layout

```
~/.titan-bos/
├── node_identity.json    # this node's keypair-derived identity
├── peers.json            # known peer registry
├── repos/                # federated bare git repos
│   ├── _manifests.json   # repo metadata
│   ├── myrepo.git/
│   └── ...
└── didn/
    ├── identities.json   # decentralized identity store
    └── data.json         # associated data records
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

All 54 tests should pass.
