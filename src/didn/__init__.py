"""
Distributed Identity & Data Network (DIDN)

A decentralized identity and data layer that replaces DNS, PKI, and SSL.

When a storage_path is provided, identities and data are persisted to disk
as JSON so they survive restarts. Without storage_path the instance is
in-memory only (useful for tests and ephemeral nodes).
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from src.crypto import verify as _verify_sig

logger = logging.getLogger(__name__)


@dataclass
class Identity:
    """Represents a decentralized identity in the network."""
    public_key: str
    signature: str
    timestamp: str
    metadata: Dict

    def to_dict(self) -> Dict:
        """Convert identity to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'Identity':
        """Create Identity from dictionary."""
        return cls(**data)


class DIDN:
    """
    Distributed Identity & Data Network implementation.

    Args:
        storage_path: Optional directory path for persistent storage.
            If None, data is kept in-memory only.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.identities: Dict[str, Identity] = {}
        self.data_store: Dict[str, Dict] = {}

        self._storage_path: Optional[Path] = None
        if storage_path:
            self._storage_path = Path(storage_path)
            self._storage_path.mkdir(parents=True, exist_ok=True)
            self._load()
            logger.info(f"DIDN loaded from {self._storage_path}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def register_identity(
        self, public_key: str, signature: str, metadata: Optional[Dict] = None
    ) -> str:
        """Register a new identity. Returns the identity ID (SHA-256 of public key)."""
        if metadata is None:
            metadata = {}

        identity_id = self._generate_identity_id(public_key)
        identity = Identity(
            public_key=public_key,
            signature=signature,
            timestamp=datetime.utcnow().isoformat(),
            metadata=metadata,
        )

        self.identities[identity_id] = identity
        self._save()
        return identity_id

    def store_data(self, identity_id: str, data: Dict, signature: str) -> str:
        """Store data under an existing identity. Returns the data ID."""
        if identity_id not in self.identities:
            raise ValueError("Unknown identity")

        data_id = self._generate_data_id(data)
        self.data_store[data_id] = {
            "data": data,
            "identity": identity_id,
            "timestamp": datetime.utcnow().isoformat(),
            "signature": signature,
        }
        self._save()
        return data_id

    def resolve_identity(self, identity_id: str) -> Optional[Identity]:
        """Resolve an identity by its ID."""
        return self.identities.get(identity_id)

    def resolve_data(self, data_id: str) -> Optional[Dict]:
        """Resolve data by its ID."""
        return self.data_store.get(data_id)

    def merge_from_peer(self, peer_identities: Dict, peer_data: Dict):
        """
        Merge identities and data received from a peer node.

        This is the replication mechanism: after connecting to a peer via QMP,
        nodes exchange their DIDN state and merge it locally.

        Rules:
        - Existing entries are NOT overwritten (local-first wins).
        - Each incoming identity is verified: identity_id must equal
          SHA-256(public_key_bytes) and the signature must be a valid
          Ed25519 signature of public_key_hex using the matching private key.
          Entries that fail either check are silently dropped with a warning.
        """
        added_identities = 0
        added_data = 0

        for iid, raw in peer_identities.items():
            if iid in self.identities:
                continue  # local-first: never overwrite

            pub_key = raw.get("public_key", "")
            signature = raw.get("signature", "")

            # 1. Verify identity_id == SHA-256(public_key_bytes)
            try:
                expected_iid = hashlib.sha256(bytes.fromhex(pub_key)).hexdigest()
            except ValueError:
                logger.warning(f"Dropping identity {iid[:16]}...: public_key is not valid hex")
                continue

            if expected_iid != iid:
                logger.warning(
                    f"Dropping identity {iid[:16]}...: "
                    "identity_id does not match SHA-256(public_key)"
                )
                continue

            # 2. Verify signature proves ownership of the private key
            if not _verify_sig(pub_key, pub_key.encode(), signature):
                logger.warning(
                    f"Dropping identity {iid[:16]}...: Ed25519 signature verification failed"
                )
                continue

            self.identities[iid] = Identity.from_dict(raw)
            added_identities += 1

        for did, raw in peer_data.items():
            if did not in self.data_store:
                self.data_store[did] = raw
                added_data += 1

        if added_identities or added_data:
            self._save()
            logger.info(
                f"Merged from peer: +{added_identities} identities, +{added_data} data entries"
            )

    def export_state(self) -> Dict:
        """Export full state for peer sync."""
        return {
            "identities": {k: v.to_dict() for k, v in self.identities.items()},
            "data": self.data_store,
        }

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _identities_file(self) -> Optional[Path]:
        return self._storage_path / "identities.json" if self._storage_path else None

    def _data_file(self) -> Optional[Path]:
        return self._storage_path / "data.json" if self._storage_path else None

    def _load(self):
        """Load state from disk."""
        ifile = self._identities_file()
        if ifile and ifile.exists():
            with open(ifile) as f:
                raw = json.load(f)
            self.identities = {k: Identity.from_dict(v) for k, v in raw.items()}

        dfile = self._data_file()
        if dfile and dfile.exists():
            with open(dfile) as f:
                self.data_store = json.load(f)

    def _save(self):
        """Persist state to disk if a storage path is configured."""
        if not self._storage_path:
            return

        ifile = self._identities_file()
        with open(ifile, "w") as f:
            json.dump({k: v.to_dict() for k, v in self.identities.items()}, f, indent=2)

        dfile = self._data_file()
        with open(dfile, "w") as f:
            json.dump(self.data_store, f, indent=2)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _generate_identity_id(self, public_key: str) -> str:
        return hashlib.sha256(public_key.encode()).hexdigest()

    def _generate_data_id(self, data: Dict) -> str:
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
