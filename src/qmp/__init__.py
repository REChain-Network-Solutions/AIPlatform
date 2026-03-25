"""
Quantum Mesh Protocol (QMP)

A protocol for communication without traditional routing, addresses, or TCP/IP.
"""

import asyncio
from typing import Dict, Any, Optional, Callable
import json
import hashlib
from dataclasses import dataclass

@dataclass
class QMPMessage:
    """Represents a message in the Quantum Mesh Protocol."""
    content: Dict[str, Any]
    sender_id: str
    message_type: str
    timestamp: float
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            'content': self.content,
            'sender_id': self.sender_id,
            'message_type': self.message_type,
            'timestamp': self.timestamp,
            'signature': self.signature
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QMPMessage':
        """Create message from dictionary."""
        return cls(**data)

class QMPService:
    """Implementation of the Quantum Mesh Protocol service."""
    
    def __init__(self, node_id: str, private_key: str = None):
        self.node_id = node_id
        self.private_key = private_key
        self.message_handlers = {}
        self.connections = set()
        self.message_queue = asyncio.Queue()
    
    async def start(self, host: str = '0.0.0.0', port: int = 0):
        """Start the QMP service."""
        self.server = await asyncio.start_server(
            self._handle_connection, 
            host=host, 
            port=port
        )
        return self.server.sockets[0].getsockname()
    
    async def stop(self):
        """Stop the QMP service."""
        self.server.close()
        await self.server.wait_closed()
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register a message handler for a specific message type."""
        self.message_handlers[message_type] = handler
    
    async def send(self, writer, message: QMPMessage):
        """Send a message to a specific connected node."""
        message_data = json.dumps(message.to_dict()).encode()
        try:
            writer.write(len(message_data).to_bytes(4, 'big') + message_data)
            await writer.drain()
        except Exception as e:
            print(f"Error sending message: {e}")
            self.connections.discard(writer)

    async def broadcast(self, message: QMPMessage, exclude: set = None):
        """Broadcast a message to all connected nodes."""
        if exclude is None:
            exclude = set()

        message_data = json.dumps(message.to_dict()).encode()
        for writer in self.connections - exclude:
            try:
                writer.write(len(message_data).to_bytes(4, 'big') + message_data)
                await writer.drain()
            except Exception as e:
                print(f"Error broadcasting message: {e}")
    
    async def _handle_connection(self, reader, writer):
        """Handle incoming connection."""
        self.connections.add(writer)
        try:
            while True:
                # Read message length (4 bytes)
                data_length = await reader.readexactly(4)
                if not data_length:
                    break
                    
                # Read message data
                data = await reader.readexactly(int.from_bytes(data_length, 'big'))
                message = QMPMessage.from_dict(json.loads(data.decode()))
                
                # Process message
                await self._process_message(message, writer)
                
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            self.connections.discard(writer)
            writer.close()
            await writer.wait_closed()
    
    async def _process_message(self, message: QMPMessage, writer):
        """Process incoming message."""
        if message.message_type in self.message_handlers:
            await self.message_handlers[message.message_type](message, writer)
        else:
            print(f"No handler for message type: {message.message_type}")
    
    def create_message(self, content: Dict, message_type: str) -> QMPMessage:
        """Create a new QMP message."""
        return QMPMessage(
            content=content,
            sender_id=self.node_id,
            message_type=message_type,
            timestamp=asyncio.get_event_loop().time()
        )
