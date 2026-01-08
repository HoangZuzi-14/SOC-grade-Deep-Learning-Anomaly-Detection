"""
Example WebSocket client for real-time streaming
"""
import asyncio
import json
import websockets
from typing import Optional


async def stream_client(uri: str = "ws://localhost:8000/api/v1/streaming/ws"):
    """Connect to WebSocket and receive real-time updates"""
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to {uri}")
            
            # Send ping
            await websocket.send(json.dumps({"type": "ping"}))
            
            # Receive messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    
                    if msg_type == "connection":
                        print(f"✓ {data.get('status')}")
                    
                    elif msg_type == "score":
                        score_data = data.get("data", {})
                        print(f"Score: {score_data.get('score', 0):.4f} | "
                              f"Severity: {score_data.get('severity')} | "
                              f"Alert: {score_data.get('alert')}")
                    
                    elif msg_type == "alert":
                        alert_data = data.get("data", {})
                        print(f"🚨 ALERT: {alert_data.get('severity')} | "
                              f"Score: {alert_data.get('score', 0):.4f} | "
                              f"ID: {alert_data.get('id')}")
                    
                    elif msg_type == "stats":
                        stats = data.get("data", {})
                        print(f"📊 Stats: Processed={stats.get('processed')} | "
                              f"Alerts={stats.get('alerts')} | "
                              f"Errors={stats.get('errors')}")
                    
                    elif msg_type == "pong":
                        print("Pong received")
                    
                    elif msg_type == "keepalive":
                        pass  # Silent keepalive
                    
                except json.JSONDecodeError:
                    print(f"Received non-JSON: {message}")
                    
    except websockets.exceptions.ConnectionClosed:
        print("Connection closed")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(stream_client())
