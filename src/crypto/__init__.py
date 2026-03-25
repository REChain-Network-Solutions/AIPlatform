"""
Cryptographic primitives for Titan-BOS.

Wraps PyNaCl Ed25519 with a simple interface used across node-registry,
DIDN, and QMP handshakes.

    from src.crypto import KeyPair, verify

    kp = KeyPair.generate()
    sig = kp.sign(b"hello")
    assert verify(kp.public_key_hex, b"hello", sig)
"""

import hashlib
import logging
from pathlib import Path

from nacl.signing import SigningKey, VerifyKey
from nacl.encoding import HexEncoder
from nacl.exceptions import BadSignatureError

logger = logging.getLogger(__name__)


class KeyPair:
    """
    Ed25519 signing keypair.

    The signing seed (private key) is 32 raw bytes stored as a hex string.
    The public key is the 32-byte Ed25519 verify key, also stored as hex.
    The node_id is SHA-256(public_key_bytes) — a 64-char hex string that
    serves as the stable identifier for a node or DIDN identity.
    """

    def __init__(self, signing_key: SigningKey):
        self._signing_key = signing_key
        self.verify_key: VerifyKey = signing_key.verify_key
        self.public_key_hex: str = signing_key.verify_key.encode(HexEncoder).decode()
        self.node_id: str = hashlib.sha256(signing_key.verify_key.encode()).hexdigest()

    # -------------------------------------------------------------------------
    # Constructors
    # -------------------------------------------------------------------------

    @classmethod
    def generate(cls) -> "KeyPair":
        """Generate a fresh random keypair."""
        return cls(SigningKey.generate())

    @classmethod
    def from_seed_hex(cls, seed_hex: str) -> "KeyPair":
        """Reconstruct a keypair from a 64-char hex seed string."""
        return cls(SigningKey(bytes.fromhex(seed_hex)))

    @classmethod
    def load_seed(cls, path: Path) -> "KeyPair":
        """Load a keypair from a seed file written by :meth:`save_seed`."""
        return cls.from_seed_hex(path.read_text().strip())

    # -------------------------------------------------------------------------
    # Operations
    # -------------------------------------------------------------------------

    def sign(self, message: bytes) -> str:
        """
        Sign *message* and return the 64-byte signature as a hex string.

        Does NOT prepend the message (unlike some NaCl wrappers).
        """
        signed = self._signing_key.sign(message)
        return signed.signature.hex()

    def seed_hex(self) -> str:
        """Export the 32-byte signing seed as a hex string (keep secret!)."""
        return self._signing_key.encode().hex()

    def save_seed(self, path: Path):
        """Write seed to *path* with mode 0o600 so only the owner can read it."""
        path.write_text(self.seed_hex())
        path.chmod(0o600)


def verify(public_key_hex: str, message: bytes, signature_hex: str) -> bool:
    """
    Verify an Ed25519 *signature_hex* over *message* using *public_key_hex*.

    Returns True if valid, False for any kind of failure (bad key, bad sig,
    wrong length, etc.) — never raises.
    """
    try:
        vk = VerifyKey(bytes.fromhex(public_key_hex))
        vk.verify(message, bytes.fromhex(signature_hex))
        return True
    except (BadSignatureError, ValueError, Exception):
        return False
