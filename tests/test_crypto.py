"""Tests for the Ed25519 crypto primitives in src/crypto."""

import pytest
from pathlib import Path
from src.crypto import KeyPair, verify


# ---------------------------------------------------------------------------
# KeyPair generation and properties
# ---------------------------------------------------------------------------

def test_generate_produces_unique_keypairs():
    kp1 = KeyPair.generate()
    kp2 = KeyPair.generate()
    assert kp1.public_key_hex != kp2.public_key_hex
    assert kp1.node_id != kp2.node_id


def test_node_id_is_sha256_of_public_key():
    import hashlib
    kp = KeyPair.generate()
    expected = hashlib.sha256(bytes.fromhex(kp.public_key_hex)).hexdigest()
    assert kp.node_id == expected


def test_public_key_hex_is_64_chars():
    kp = KeyPair.generate()
    assert len(kp.public_key_hex) == 64   # 32 bytes = 64 hex chars


def test_node_id_is_64_chars():
    kp = KeyPair.generate()
    assert len(kp.node_id) == 64


# ---------------------------------------------------------------------------
# Sign / verify round-trip
# ---------------------------------------------------------------------------

def test_sign_verify_roundtrip():
    kp = KeyPair.generate()
    msg = b"hello titan-bos"
    sig = kp.sign(msg)
    assert verify(kp.public_key_hex, msg, sig)


def test_verify_fails_wrong_message():
    kp = KeyPair.generate()
    sig = kp.sign(b"correct message")
    assert not verify(kp.public_key_hex, b"wrong message", sig)


def test_verify_fails_wrong_key():
    kp1 = KeyPair.generate()
    kp2 = KeyPair.generate()
    sig = kp1.sign(b"data")
    assert not verify(kp2.public_key_hex, b"data", sig)


def test_verify_fails_tampered_signature():
    kp = KeyPair.generate()
    sig = kp.sign(b"data")
    bad_sig = "ff" * 64  # 64 bytes of garbage
    assert not verify(kp.public_key_hex, b"data", bad_sig)


def test_verify_fails_bad_hex_key():
    assert not verify("not-hex", b"data", "aa" * 64)


def test_verify_fails_bad_hex_signature():
    kp = KeyPair.generate()
    assert not verify(kp.public_key_hex, b"data", "not-hex")


def test_signature_is_hex_string():
    kp = KeyPair.generate()
    sig = kp.sign(b"test")
    assert isinstance(sig, str)
    bytes.fromhex(sig)  # must be valid hex


def test_signature_is_128_chars():
    """Ed25519 signatures are 64 bytes = 128 hex chars."""
    kp = KeyPair.generate()
    sig = kp.sign(b"test")
    assert len(sig) == 128


# ---------------------------------------------------------------------------
# Seed export / import
# ---------------------------------------------------------------------------

def test_from_seed_hex_reconstructs_keypair():
    kp = KeyPair.generate()
    kp2 = KeyPair.from_seed_hex(kp.seed_hex())
    assert kp2.public_key_hex == kp.public_key_hex
    assert kp2.node_id == kp.node_id


def test_seed_hex_is_64_chars():
    kp = KeyPair.generate()
    assert len(kp.seed_hex()) == 64


def test_reconstructed_keypair_produces_verifiable_signature():
    kp = KeyPair.generate()
    kp2 = KeyPair.from_seed_hex(kp.seed_hex())
    sig = kp2.sign(b"message")
    assert verify(kp.public_key_hex, b"message", sig)


# ---------------------------------------------------------------------------
# Seed file I/O
# ---------------------------------------------------------------------------

def test_save_and_load_seed(tmp_path):
    kp = KeyPair.generate()
    seed_file = tmp_path / "keypair.seed"
    kp.save_seed(seed_file)

    kp2 = KeyPair.load_seed(seed_file)
    assert kp2.public_key_hex == kp.public_key_hex


def test_seed_file_has_restrictive_permissions(tmp_path):
    kp = KeyPair.generate()
    seed_file = tmp_path / "keypair.seed"
    kp.save_seed(seed_file)
    mode = seed_file.stat().st_mode & 0o777
    assert mode == 0o600


# ---------------------------------------------------------------------------
# DIDN handshake pattern (sign your own public_key_hex)
# ---------------------------------------------------------------------------

def test_didn_self_signature_pattern():
    """
    The DIDN identity registration pattern: sign your own public_key_hex to
    prove you hold the private key. This is the standard proof-of-ownership.
    """
    kp = KeyPair.generate()
    sig = kp.sign(kp.public_key_hex.encode())
    assert verify(kp.public_key_hex, kp.public_key_hex.encode(), sig)


def test_node_handshake_signature_pattern():
    """
    QMP handshake pattern: sign node_id (= SHA-256(pub_key)) to prove ownership.
    """
    kp = KeyPair.generate()
    sig = kp.sign(kp.node_id.encode())
    assert verify(kp.public_key_hex, kp.node_id.encode(), sig)
