# Real-time Streaming Processing

## Overview

Real-time log streaming and inference pipeline với WebSocket support cho live updates.

## Features

- **Log Streaming**: Real-time log file monitoring (tail -f style)
- **Streaming Inference**: Real-time anomaly detection
- **WebSocket Support**: Live updates to dashboard
- **Alert Broadcasting**: Automatic alert notifications
- **Statistics**: Real-time processing stats

## Architecture

```
Log File → LogStreamer → LogProcessor → StreamingInference → WebSocket → Dashboard
```

## Components

### 1. LogStreamer

Stream logs from file in real-time:

```python
from streaming.log_streamer import LogStreamer

streamer = LogStreamer("path/to/logfile.log")
async for line in streamer.stream():
    print(line)
```

### 2. LogProcessor

Process log stream and extract sequences:

```python
from streaming.log_streamer import LogProcessor

processor = LogProcessor()
processor.set_window_size(5)

async for seq_data in processor.process_stream(streamer.stream()):
    print(seq_data["sequence"])
```

### 3. StreamingInference

Real-time inference on sequences:

```python
from streaming.streaming_inference import StreamingInference

inference = StreamingInference(model, score_func)
async for result in inference.process_sequences(sequence_stream):
    print(result["score"], result["alert"])
```

### 4. WebSocket Handler

Broadcast updates to connected clients:

```python
from streaming.websocket_handler import broadcast_score, broadcast_alert

await broadcast_score(score_result)
await broadcast_alert(alert_data)
```

## API Endpoints

### WebSocket Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/streaming/ws');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data);
};
```

### Start Streaming

```bash
POST /api/v1/streaming/start
Body: {
    "log_file": "/path/to/logfile.log"
}
```

### Stop Streaming

```bash
POST /api/v1/streaming/stop
```

### Get Status

```bash
GET /api/v1/streaming/status
```

### Upload Log File

```bash
POST /api/v1/streaming/upload
Body: {
    "log_file": "/path/to/logfile.log"
}
```

## WebSocket Message Types

### Connection

```json
{
    "type": "connection",
    "status": "connected",
    "timestamp": "2024-01-01T12:00:00"
}
```

### Score Update

```json
{
    "type": "score",
    "data": {
        "score": 8.5,
        "severity": "HIGH",
        "alert": true,
        "sequence": [1, 2, 3, 4, 5],
        "timestamp": "2024-01-01T12:00:00"
    },
    "timestamp": "2024-01-01T12:00:00"
}
```

### Alert

```json
{
    "type": "alert",
    "data": {
        "id": 123,
        "severity": "HIGH",
        "score": 8.5,
        "sequence": [1, 2, 3, 4, 5],
        "created_at": "2024-01-01T12:00:00"
    },
    "timestamp": "2024-01-01T12:00:00"
}
```

### Statistics

```json
{
    "type": "stats",
    "data": {
        "processed": 1000,
        "alerts": 25,
        "errors": 2,
        "throughput": 10.5
    },
    "timestamp": "2024-01-01T12:00:00"
}
```

## Usage Examples

### Python Client

```python
import asyncio
from streaming.client_example import stream_client

asyncio.run(stream_client())
```

### JavaScript Client

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/streaming/ws');

ws.onopen = () => {
    console.log('Connected');
    ws.send(JSON.stringify({type: 'ping'}));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'alert') {
        console.log('Alert!', data.data);
    } else if (data.type === 'score') {
        console.log('Score:', data.data.score);
    }
};
```

### Start Streaming via API

```python
import requests

# Start streaming
response = requests.post(
    'http://localhost:8000/api/v1/streaming/start',
    json={"log_file": "/var/log/auth.log"}
)
print(response.json())

# Check status
response = requests.get('http://localhost:8000/api/v1/streaming/status')
print(response.json())
```

## Integration with Dashboard

Dashboard có thể kết nối WebSocket để nhận real-time updates:

```javascript
// In dashboard component
useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/api/v1/streaming/ws');
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'alert') {
            setAlerts(prev => [data.data, ...prev]);
        } else if (data.type === 'score') {
            updateScoreChart(data.data);
        }
    };
    
    return () => ws.close();
}, []);
```

## Performance

- **Throughput**: ~10-100 sequences/second (depends on model complexity)
- **Latency**: <100ms per sequence
- **Memory**: Minimal (streaming, not batch processing)

## Error Handling

- Automatic reconnection on WebSocket disconnect
- Error logging for failed sequences
- Graceful degradation if model unavailable

## Next Steps

- ✅ Streaming pipeline complete
- ⏭️ Dashboard WebSocket integration
- ⏭️ Performance optimization
- ⏭️ Multi-file streaming support
