"""
Streaming API routes for real-time processing
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Optional
import json
import asyncio
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from streaming.websocket_handler import (
    websocket_endpoint,
    manager,
    broadcast_score,
    broadcast_alert,
    broadcast_stats
)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from streaming.log_streamer import LogStreamer, LogProcessor
from streaming.streaming_inference import StreamingInference
from api.database import get_db, create_alert_from_score
from sqlalchemy.orm import Session

router = APIRouter(prefix="/api/v1/streaming", tags=["streaming"])


# Global streaming state
streaming_task = None
streaming_active = False


@router.websocket("/ws")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming"""
    await websocket_endpoint(websocket)


@router.post("/start")
async def start_streaming(
    log_file: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start real-time log streaming and inference"""
    global streaming_task, streaming_active
    
    if streaming_active:
        return {"status": "already_running", "message": "Streaming already active"}
    
    MODEL, MODEL_CONFIG, THRESHOLDS, score_sequence = get_model_and_config()
    
    if MODEL is None:
        return {"status": "error", "message": "Model not loaded"}
    
    # Create streamer
    streamer = LogStreamer(log_file)
    processor = LogProcessor()
    processor.set_window_size(MODEL_CONFIG.get("window_size", 5))
    
    # Create inference
    def score_func(sequence: List[int]):
        """Scoring function"""
        result = score_sequence(sequence, "lstm", db, store_alert=False)
        return result
    
    inference = StreamingInference(MODEL, score_func)
    
    # Start streaming task
    async def stream_loop():
        global streaming_active
        streaming_active = True
        
        try:
            async for seq_data in processor.process_stream(streamer.stream()):
                async for score_result in inference.process_sequences([seq_data]):
                    # Broadcast score
                    await broadcast_score(score_result)
                    
                    # If alert, create in database and broadcast
                    if score_result.get("alert"):
                        try:
                            alert = create_alert_from_score(
                                db=db,
                                sequence=score_result["sequence"],
                                score=score_result["score"],
                                severity=score_result["severity"]
                            )
                            await broadcast_alert({
                                "id": alert.id,
                                "severity": alert.severity,
                                "score": alert.score,
                                "sequence": alert.sequence,
                                "created_at": alert.created_at.isoformat()
                            })
                        except Exception as e:
                            print(f"Error creating alert: {e}")
                    
                    # Broadcast stats periodically
                    if inference.stats["processed"] % 10 == 0:
                        await broadcast_stats(inference.get_stats())
                        
        except Exception as e:
            print(f"Streaming error: {e}")
        finally:
            streaming_active = False
    
    streaming_task = asyncio.create_task(stream_loop())
    
    return {
        "status": "started",
        "log_file": log_file,
        "message": "Streaming started"
    }


@router.post("/stop")
async def stop_streaming():
    """Stop real-time streaming"""
    global streaming_task, streaming_active
    
    if not streaming_active:
        return {"status": "not_running", "message": "Streaming not active"}
    
    if streaming_task:
        streaming_task.cancel()
        try:
            await streaming_task
        except asyncio.CancelledError:
            pass
    
    streaming_active = False
    
    return {"status": "stopped", "message": "Streaming stopped"}


@router.get("/status")
async def get_streaming_status():
    """Get streaming status"""
    return {
        "active": streaming_active,
        "connections": manager.get_connection_count(),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/upload")
async def upload_log_stream(
    log_file: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Upload and process log file in streaming fashion
    """
    MODEL, MODEL_CONFIG, THRESHOLDS, score_sequence = get_model_and_config()
    
    if MODEL is None:
        return {"status": "error", "message": "Model not loaded"}
    
    # Create streamer for uploaded file
    streamer = LogStreamer(log_file, delay=0.01)  # Faster for file upload
    processor = LogProcessor()
    window_size = MODEL_CONFIG.get("window_size", 5) if MODEL_CONFIG else 5
    processor.set_window_size(window_size)
    
    def score_func(sequence: List[int]):
        return score_sequence(sequence, "lstm", db, store_alert=True)
    
    inference = StreamingInference(MODEL, score_func)
    
    async def process_upload():
        """Process uploaded log file"""
        results = []
        alerts = []
        
        async for seq_data in processor.process_stream(streamer.stream()):
            async for score_result in inference.process_sequences([seq_data]):
                results.append(score_result)
                
                if score_result.get("alert"):
                    alerts.append(score_result)
                    # Create alert in database
                    try:
                        create_alert_from_score(
                            db=db,
                            sequence=score_result["sequence"],
                            score=score_result["score"],
                            severity=score_result["severity"]
                        )
                    except Exception as e:
                        print(f"Error creating alert: {e}")
        
        return {
            "processed": len(results),
            "alerts": len(alerts),
            "stats": inference.get_stats()
        }
    
    # Run in background
    background_tasks.add_task(process_upload)
    
    return {
        "status": "processing",
        "message": "Log file is being processed",
        "log_file": log_file
    }
