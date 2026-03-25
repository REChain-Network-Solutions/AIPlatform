"""Tests for the Quantum Mesh Protocol (QMP) component."""

import asyncio
import json
import pytest
from src.qmp import QMPService, QMPMessage
from unittest.mock import AsyncMock, MagicMock

# Test data
TEST_MESSAGE = {
    "content": {"text": "Hello, QMP!"},
    "message_type": "test_message",
    "sender_id": "test_sender",
    "timestamp": 1730370000.0,
    "signature": None
}

@pytest.fixture
def qmp_service():
    """Fixture providing a QMPService instance for testing."""
    return QMPService("test_node")

@pytest.fixture
def mock_writer():
    """Fixture providing a mock writer for testing."""
    writer = AsyncMock()
    writer.drain = AsyncMock()
    return writer

@pytest.mark.asyncio
async def test_message_serialization():
    """Test message serialization and deserialization."""
    # Create a message
    original = QMPMessage(
        content={"text": "Hello"},
        sender_id="test",
        message_type="test",
        timestamp=1234567890.0
    )
    
    # Convert to dict and back
    data = original.to_dict()
    restored = QMPMessage.from_dict(data)
    
    # Verify data integrity
    assert restored.content == original.content
    assert restored.sender_id == original.sender_id
    assert restored.message_type == original.message_type
    assert restored.timestamp == original.timestamp

@pytest.mark.asyncio
async def test_message_handler_registration(qmp_service):
    """Test registering and triggering message handlers."""
    # Create a mock handler
    mock_handler = AsyncMock()
    
    # Register the handler
    qmp_service.register_handler("test_message", mock_handler)
    
    # Create a test message
    message = QMPMessage.from_dict(TEST_MESSAGE)
    
    # Process the message
    mock_writer = AsyncMock()
    await qmp_service._process_message(message, mock_writer)
    
    # Verify the handler was called
    mock_handler.assert_awaited_once()
    
    # Check the handler received the correct arguments
    args, _ = mock_handler.await_args
    assert args[0] == message
    assert args[1] == mock_writer

@pytest.mark.asyncio
async def test_broadcast_message(qmp_service, mock_writer):
    """Test broadcasting a message to connected nodes."""
    # Add mock writers to connections
    qmp_service.connections.add(mock_writer)
    
    # Create a test message
    message = QMPMessage.from_dict(TEST_MESSAGE)
    
    # Broadcast the message
    await qmp_service.broadcast(message)
    
    # Verify the message was written
    mock_writer.write.assert_called_once()

    # Get the actual data that was written
    call_args = mock_writer.write.call_args[0][0]

    # First 4 bytes are the payload length
    data_length = int.from_bytes(call_args[:4], 'big')
    payload = call_args[4:]
    assert data_length == len(payload)

    # Decode and compare as parsed JSON (order-independent)
    actual = json.loads(payload.decode())
    assert actual["content"] == TEST_MESSAGE["content"]
    assert actual["sender_id"] == TEST_MESSAGE["sender_id"]
    assert actual["message_type"] == TEST_MESSAGE["message_type"]
    assert actual["timestamp"] == TEST_MESSAGE["timestamp"]
    mock_writer.drain.assert_awaited_once()

@pytest.mark.asyncio
async def test_connection_handling(qmp_service, mock_writer):
    """Test handling of incoming connections and messages."""
    # Create a mock reader
    mock_reader = AsyncMock()
    
    # Create a test message
    message_data = json.dumps(TEST_MESSAGE).encode()
    message_length = len(message_data).to_bytes(4, 'big')
    
    # Configure the reader to return the message
    mock_reader.readexactly.side_effect = [
        message_length,  # First read: message length
        message_data,    # Second read: message data
        b''              # Third read: empty data to end the loop
    ]
    
    # Register a mock handler
    mock_handler = AsyncMock()
    qmp_service.register_handler("test_message", mock_handler)
    
    # Process the connection
    await qmp_service._handle_connection(mock_reader, mock_writer)
    
    # Verify the connection was added and then removed
    assert mock_writer not in qmp_service.connections
    
    # Verify the handler was called with the correct message
    mock_handler.assert_awaited_once()
    args, _ = mock_handler.await_args
    assert args[0].content == TEST_MESSAGE["content"]
    assert args[1] == mock_writer

@pytest.mark.asyncio
async def test_create_message(qmp_service):
    """Test creating a new QMP message."""
    content = {"text": "Hello, world!"}
    message = qmp_service.create_message(content, "test_type")
    
    assert message.content == content
    assert message.message_type == "test_type"
    assert message.sender_id == "test_node"
    assert isinstance(message.timestamp, float)
    assert message.signature is None
