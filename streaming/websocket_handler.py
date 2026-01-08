"""
WebSocket handler for real-time streaming
"""
import json
import asyncio
from typing import Set, Optional
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime


class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections.discard(websocket)
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific connection"""
        try:
            await websocket.send_json(message)
        except:
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connections"""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.add(connection)
        
        # Remove disconnected
        for conn in disconnected:
            self.disconnect(conn)
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint handler"""
    await manager.connect(websocket)
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for client message (ping/pong or commands)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                try:
                    message = json.loads(data)
                    # Handle client commands
                    if message.get("type") == "ping":
                        await websocket.send_json({
                            "type": "pong",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                except json.JSONDecodeError:
                    pass
                    
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({
                    "type": "keepalive",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def broadcast_score(score_result: dict):
    """Broadcast score result to all connected clients"""
    message = {
        "type": "score",
        "data": score_result,
        "timestamp": datetime.utcnow().isoformat()
    }
    await manager.broadcast(message)


async def broadcast_alert(alert_data: dict):
    """Broadcast alert to all connected clients"""
    message = {
        "type": "alert",
        "data": alert_data,
        "timestamp": datetime.utcnow().isoformat()
    }
    await manager.broadcast(message)


async def broadcast_stats(stats: dict):
    """Broadcast statistics to all connected clients"""
    message = {
        "type": "stats",
        "data": stats,
        "timestamp": datetime.utcnow().isoformat()
    }
    await manager.broadcast(message)
