"""
REST API for SOC Log Anomaly Detection System
FastAPI-based API for real-time inference and alert management
"""
from __future__ import annotations

import os
import pickle
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from model.deeplog_lstm import DeepLogLSTM
from scoring.anomaly_score import compute_thresholds, score_to_soc_decision, Thresholds
from api.database import (
    get_db, init_db, create_alert_from_score, get_recent_alerts,
    update_alert_status, get_alert_statistics,
    LogEntry, Sequence, AnomalyScore, Alert, ModelRun
)
from api.streaming_routes import router as streaming_router
from api.explainability_routes import router as explainability_router
from api.prioritization_routes import router as prioritization_router

# Initialize FastAPI app
app = FastAPI(
    title="SOC Log Anomaly Detection API",
    description="REST API for log anomaly detection using Deep Learning",
    version="1.0.0"
)

# CORS middleware for web dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and config
MODEL = None
MODEL_CONFIG = {}
THRESHOLDS = None
EVENT_MAPPING = {}


# Pydantic models for request/response
class SequenceRequest(BaseModel):
    sequence: List[int] = Field(..., description="Log template ID sequence")
    model_type: str = Field(default="lstm", description="Model type: 'lstm' or 'if'")


class BatchSequenceRequest(BaseModel):
    sequences: List[List[int]] = Field(..., description="List of log template ID sequences")
    model_type: str = Field(default="lstm", description="Model type: 'lstm' or 'if'")


class ScoreResponse(BaseModel):
    score: float
    severity: str
    alert: bool
    sequence: List[int]
    prediction: Optional[List[int]] = None
    alert_id: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    config: Dict[str, Any]  # Renamed from model_config (Pydantic v2 reserved keyword)
    thresholds: Optional[Dict[str, float]] = None
    database_connected: bool


class AlertRequest(BaseModel):
    sequence: List[int]
    score: float
    severity: str
    timestamp: Optional[str] = None


class AlertUpdateRequest(BaseModel):
    status: str
    user: Optional[str] = None


class AlertResponse(BaseModel):
    id: int
    severity: str
    score: float
    status: str
    priority: int
    sequence: List[int]
    created_at: str
    updated_at: str


# Model loading functions
def load_model(model_path: str, device: str = "cpu") -> tuple:
    """Load trained LSTM model from checkpoint"""
    global MODEL, MODEL_CONFIG
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt["model_state_dict"]
    
    num_labels = int(ckpt["num_labels"])
    window_size = int(ckpt["window_size"])
    embedding_dim = state_dict["embedding.weight"].shape[1]
    hidden_size = state_dict["lstm.weight_hh_l0"].shape[1]
    num_layers = 1
    
    model = DeepLogLSTM(
        num_labels=num_labels,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(device)
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    MODEL_CONFIG = {
        "num_labels": num_labels,
        "window_size": window_size,
        "embedding_dim": embedding_dim,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "device": device
    }
    
    return model, MODEL_CONFIG


def load_thresholds(scores_path: str) -> Thresholds:
    """Load precomputed thresholds from scores file"""
    global THRESHOLDS
    
    if not os.path.exists(scores_path):
        return None
    
    with open(scores_path, "rb") as f:
        scores = pickle.load(f)
    
    THRESHOLDS = compute_thresholds(scores)
    return THRESHOLDS


def load_event_mapping(mapping_path: str) -> Dict:
    """Load event ID to template text mapping"""
    global EVENT_MAPPING
    
    if not os.path.exists(mapping_path):
        return {}
    
    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle different mapping formats
    if isinstance(data, dict):
        if "id_to_template" in data:
            EVENT_MAPPING = data["id_to_template"]
        elif "idx2event" in data:
            EVENT_MAPPING = data["idx2event"]
        else:
            EVENT_MAPPING = data
    else:
        EVENT_MAPPING = {}
    
    return EVENT_MAPPING


# Inference functions
@torch.no_grad()
def score_sequence(sequence: List[int], model_type: str = "lstm", db: Optional[Session] = None) -> Dict[str, Any]:
    """Score a single sequence for anomaly detection"""
    global MODEL, MODEL_CONFIG, THRESHOLDS
    
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not sequence or len(sequence) < 2:
        return {
            "score": 0.0,
            "severity": "NONE",
            "alert": False,
            "sequence": sequence
        }
    
    window_size = MODEL_CONFIG["window_size"]
    device = MODEL_CONFIG["device"]
    
    # Prepare sequence
    if len(sequence) > window_size:
        seq = sequence[-window_size:]
    elif len(sequence) < window_size:
        pad_len = window_size - len(sequence)
        seq = ([0] * pad_len) + sequence
    else:
        seq = sequence
    
    # Create input tensor
    x = torch.tensor(seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(seq[-1], dtype=torch.long, device=device)
    
    # Get model prediction
    logits = MODEL(x).squeeze(0)
    logp = F.log_softmax(logits, dim=-1)
    score = float((-logp[y]).item())
    
    # Get top predictions
    top_k = 5
    top_probs, top_indices = torch.topk(F.softmax(logits, dim=-1), k=min(top_k, logits.numel()))
    predictions = [
        {"template_id": int(idx), "probability": float(prob)}
        for idx, prob in zip(top_indices, top_probs)
    ]
    
    # Compute severity and alert
    if THRESHOLDS:
        alert, severity = score_to_soc_decision(score, THRESHOLDS, min_alert="MED")
    else:
        severity = "UNKNOWN"
        alert = False
    
    result = {
        "score": score,
        "severity": severity,
        "alert": alert,
        "sequence": sequence,
        "predictions": predictions
    }
    
    # Store in database if alert and db provided
    if alert and db and store_alert:
        try:
            # Store sequence
            db_sequence = Sequence(sequence=sequence, window_size=len(sequence))
            db.add(db_sequence)
            db.commit()
            db.refresh(db_sequence)
            
            # Store score
            db_score = AnomalyScore(
                sequence_id=db_sequence.id,
                score=score,
                severity=severity,
                model_type=model_type,
                predictions=predictions
            )
            db.add(db_score)
            db.commit()
            db.refresh(db_score)
            
            # Create alert
            db_alert = create_alert_from_score(
                db=db,
                sequence=sequence,
                score=score,
                severity=severity,
                score_id=db_score.id,
                sequence_id=db_sequence.id,
                predictions=predictions
            )
            
            result["alert_id"] = db_alert.id
        except Exception as e:
            print(f"[!] Error storing alert in database: {e}")
            # Continue without database storage
    
    return result


# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root(db: Session = Depends(get_db)):
    """Root endpoint - health check"""
    db_connected = False
    try:
        db.execute("SELECT 1")
        db_connected = True
    except:
        pass
    
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "config": MODEL_CONFIG if MODEL_CONFIG else {},  # Renamed from model_config
        "thresholds": {
            "p95": THRESHOLDS.p95,
            "p99": THRESHOLDS.p99,
            "p999": THRESHOLDS.p999
        } if THRESHOLDS else None,
        "database_connected": db_connected
    }


@app.post("/api/v1/score", response_model=ScoreResponse)
async def score_single(request: SequenceRequest, db: Session = Depends(get_db)):
    """Score a single log sequence for anomalies"""
    try:
        result = score_sequence(request.sequence, request.model_type, db)
        return ScoreResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/score/batch")
async def score_batch(request: BatchSequenceRequest, db: Session = Depends(get_db)):
    """Score multiple sequences in batch"""
    try:
        results = []
        for seq in request.sequences:
            result = score_sequence(seq, request.model_type, db)
            results.append(result)
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/model/info")
async def model_info():
    """Get model information and configuration"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_loaded": True,
        "config": MODEL_CONFIG,  # Using "config" instead of "model_config"
        "thresholds": {
            "p95": THRESHOLDS.p95,
            "p99": THRESHOLDS.p99,
            "p999": THRESHOLDS.p999
        } if THRESHOLDS else None,
        "event_mapping_count": len(EVENT_MAPPING)
    }


@app.get("/api/v1/events/{event_id}")
async def get_event_info(event_id: int):
    """Get event template information by ID"""
    if event_id in EVENT_MAPPING:
        return {
            "event_id": event_id,
            "template": EVENT_MAPPING[event_id]
        }
    else:
        raise HTTPException(status_code=404, detail=f"Event ID {event_id} not found")


@app.post("/api/v1/alerts")
async def create_alert(alert: AlertRequest, db: Session = Depends(get_db)):
    """Create a new alert"""
    try:
        db_alert = create_alert_from_score(
            db=db,
            sequence=alert.sequence,
            score=alert.score,
            severity=alert.severity
        )
        return {
            "id": db_alert.id,
            "sequence": db_alert.sequence,
            "score": db_alert.score,
            "severity": db_alert.severity,
            "status": db_alert.status,
            "created_at": db_alert.created_at.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/alerts", response_model=List[AlertResponse])
async def list_alerts(
    limit: int = 100,
    severity: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List alerts with optional filters"""
    alerts = get_recent_alerts(db, limit=limit, severity=severity, status=status)
    return [
        AlertResponse(
            id=alert.id,
            severity=alert.severity,
            score=alert.score,
            status=alert.status,
            priority=alert.priority,
            sequence=alert.sequence,
            created_at=alert.created_at.isoformat(),
            updated_at=alert.updated_at.isoformat() if alert.updated_at else alert.created_at.isoformat()
        )
        for alert in alerts
    ]


@app.get("/api/v1/alerts/{alert_id}", response_model=AlertResponse)
async def get_alert(alert_id: int, db: Session = Depends(get_db)):
    """Get a specific alert by ID"""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    
    return AlertResponse(
        id=alert.id,
        severity=alert.severity,
        score=alert.score,
        status=alert.status,
        priority=alert.priority,
        sequence=alert.sequence,
        created_at=alert.created_at.isoformat(),
        updated_at=alert.updated_at.isoformat() if alert.updated_at else alert.created_at.isoformat()
    )


@app.patch("/api/v1/alerts/{alert_id}")
async def update_alert(
    alert_id: int,
    update: AlertUpdateRequest,
    db: Session = Depends(get_db)
):
    """Update alert status"""
    alert = update_alert_status(db, alert_id, update.status, update.user)
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    
    return {
        "id": alert.id,
        "status": alert.status,
        "updated_at": alert.updated_at.isoformat() if alert.updated_at else alert.created_at.isoformat()
    }


@app.get("/api/v1/alerts/statistics/summary")
async def alert_statistics(db: Session = Depends(get_db)):
    """Get alert statistics"""
    stats = get_alert_statistics(db)
    return stats


@app.post("/api/v1/logs")
async def create_log_entry(
    template_id: int,
    template_text: Optional[str] = None,
    raw_log: Optional[str] = None,
    parameters: Optional[Dict] = None,
    source: Optional[str] = "unknown",
    db: Session = Depends(get_db)
):
    """Create a new log entry"""
    log_entry = LogEntry(
        template_id=template_id,
        template_text=template_text,
        raw_log=raw_log,
        parameters=parameters or {},
        source=source
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)
    return {
        "id": log_entry.id,
        "template_id": log_entry.template_id,
        "timestamp": log_entry.timestamp.isoformat(),
        "source": log_entry.source
    }


@app.get("/api/v1/logs")
async def list_logs(
    limit: int = 100,
    source: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List log entries"""
    query = db.query(LogEntry)
    if source:
        query = query.filter(LogEntry.source == source)
    
    logs = query.order_by(LogEntry.timestamp.desc()).limit(limit).all()
    return [
        {
            "id": log.id,
            "template_id": log.template_id,
            "template_text": log.template_text,
            "timestamp": log.timestamp.isoformat(),
            "source": log.source
        }
        for log in logs
    ]


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load model and configurations on startup"""
    global MODEL, THRESHOLDS, EVENT_MAPPING
    
    # Initialize database
    init_db()
    
    # Default paths (can be overridden by environment variables)
    project_root = Path(__file__).parent.parent
    model_path = os.getenv("MODEL_PATH", str(project_root / "model" / "model.pth"))
    scores_path = os.getenv("SCORES_PATH", str(project_root / "data" / "sequences" / "lstm_scores.pkl"))
    mapping_path = os.getenv("MAPPING_PATH", str(project_root / "data" / "sequences" / "event_mapping.json"))
    device = os.getenv("DEVICE", "cpu")
    
    try:
        print(f"[*] Loading model from {model_path}...")
        MODEL, _ = load_model(model_path, device)
        print(f"[+] Model loaded successfully")
        
        print(f"[*] Loading thresholds from {scores_path}...")
        THRESHOLDS = load_thresholds(scores_path)
        if THRESHOLDS:
            print(f"[+] Thresholds loaded: p95={THRESHOLDS.p95:.4f}, p99={THRESHOLDS.p99:.4f}, p999={THRESHOLDS.p999:.4f}")
        else:
            print(f"[!] Thresholds not found, using default")
        
        print(f"[*] Loading event mapping from {mapping_path}...")
        EVENT_MAPPING = load_event_mapping(mapping_path)
        print(f"[+] Event mapping loaded: {len(EVENT_MAPPING)} events")
        
    except Exception as e:
        print(f"[!] Error loading model: {e}")
        print(f"[!] API will start but model endpoints will be unavailable")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
