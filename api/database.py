"""
Database models and configuration for SOC Anomaly Detection System
Supports both SQLite (development) and PostgreSQL (production)
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, Float, String, Text, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

# Base class for models
Base = declarative_base()


# Database Models
class LogEntry(Base):
    """Log entry model"""
    __tablename__ = "log_entries"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    template_id = Column(Integer, index=True)
    template_text = Column(Text)
    raw_log = Column(Text)
    parameters = Column(JSON)  # Store extracted parameters as JSON
    source = Column(String(100))  # e.g., "auth.log", "system.log"
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<LogEntry(id={self.id}, template_id={self.template_id}, timestamp={self.timestamp})>"


class Sequence(Base):
    """Log sequence model"""
    __tablename__ = "sequences"

    id = Column(Integer, primary_key=True, index=True)
    sequence = Column(JSON)  # List of template IDs
    window_size = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<Sequence(id={self.id}, length={len(self.sequence) if self.sequence else 0})>"


class AnomalyScore(Base):
    """Anomaly score model"""
    __tablename__ = "anomaly_scores"

    id = Column(Integer, primary_key=True, index=True)
    sequence_id = Column(Integer, index=True)  # Reference to Sequence
    score = Column(Float, index=True)
    severity = Column(String(20), index=True)  # NONE, LOW, MED, HIGH
    model_type = Column(String(20))  # "lstm", "if", etc.
    predictions = Column(JSON)  # Top-k predictions
    computed_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<AnomalyScore(id={self.id}, score={self.score:.4f}, severity={self.severity})>"


class Alert(Base):
    """Alert model"""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    sequence_id = Column(Integer, index=True)  # Reference to Sequence
    score_id = Column(Integer, index=True)  # Reference to AnomalyScore
    severity = Column(String(20), index=True)  # NONE, LOW, MED, HIGH
    score = Column(Float)
    sequence = Column(JSON)  # Store sequence for quick access
    status = Column(String(20), default="OPEN")  # OPEN, ACKNOWLEDGED, RESOLVED, FALSE_POSITIVE
    priority = Column(Integer, default=0)  # Calculated priority score
    description = Column(Text)
    acknowledged_by = Column(String(100))
    acknowledged_at = Column(DateTime)
    resolved_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Alert(id={self.id}, severity={self.severity}, status={self.status})>"


class ModelRun(Base):
    """Model training/inference run tracking"""
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String(50))
    model_path = Column(String(500))
    config = Column(JSON)  # Model configuration
    metrics = Column(JSON)  # Training/inference metrics
    status = Column(String(20))  # TRAINING, COMPLETED, FAILED
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    error_message = Column(Text)

    def __repr__(self):
        return f"<ModelRun(id={self.id}, model_type={self.model_type}, status={self.status})>"


# Database configuration
def get_database_url() -> str:
    """Get database URL from environment or use default SQLite"""
    db_url = os.getenv("DATABASE_URL")
    
    if db_url:
        return db_url
    
    # Default to SQLite in project root
    project_root = Path(__file__).parent.parent
    db_path = project_root / "data" / "soc_anomaly.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    return f"sqlite:///{db_path}"


def create_engine_with_config():
    """Create database engine with appropriate configuration"""
    db_url = get_database_url()
    
    if db_url.startswith("sqlite"):
        # SQLite configuration
        return create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=os.getenv("DB_ECHO", "false").lower() == "true"
        )
    else:
        # PostgreSQL or other database
        return create_engine(
            db_url,
            pool_pre_ping=True,
            echo=os.getenv("DB_ECHO", "false").lower() == "true"
        )


# Create engine and session
engine = create_engine_with_config()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database - create all tables"""
    Base.metadata.create_all(bind=engine)
    print(f"[+] Database initialized: {get_database_url()}")


def get_db() -> Session:
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Database utility functions
def create_alert_from_score(
    db: Session,
    sequence: list[int],
    score: float,
    severity: str,
    score_id: Optional[int] = None,
    sequence_id: Optional[int] = None,
    predictions: Optional[list] = None
) -> Alert:
    """Create an alert from anomaly score"""
    # Calculate priority based on severity and score
    priority_map = {"HIGH": 100, "MED": 50, "LOW": 25, "NONE": 0}
    priority = priority_map.get(severity, 0) + int(score * 10)
    
    alert = Alert(
        sequence_id=sequence_id,
        score_id=score_id,
        severity=severity,
        score=score,
        sequence=sequence,
        priority=priority,
        status="OPEN" if severity in ["HIGH", "MED"] else "OPEN",
        description=f"Anomaly detected with {severity} severity (score: {score:.4f})"
    )
    
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


def get_recent_alerts(
    db: Session,
    limit: int = 100,
    severity: Optional[str] = None,
    status: Optional[str] = None
) -> list[Alert]:
    """Get recent alerts with optional filters"""
    query = db.query(Alert)
    
    if severity:
        query = query.filter(Alert.severity == severity)
    
    if status:
        query = query.filter(Alert.status == status)
    
    return query.order_by(Alert.created_at.desc()).limit(limit).all()


def update_alert_status(
    db: Session,
    alert_id: int,
    status: str,
    user: Optional[str] = None
) -> Optional[Alert]:
    """Update alert status"""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    
    if not alert:
        return None
    
    alert.status = status
    alert.updated_at = datetime.utcnow()
    
    if status == "ACKNOWLEDGED":
        alert.acknowledged_by = user
        alert.acknowledged_at = datetime.utcnow()
    elif status == "RESOLVED":
        alert.resolved_at = datetime.utcnow()
    
    db.commit()
    db.refresh(alert)
    return alert


def get_alert_statistics(db: Session) -> dict:
    """Get alert statistics"""
    total = db.query(Alert).count()
    by_severity = {}
    by_status = {}
    
    for severity in ["HIGH", "MED", "LOW", "NONE"]:
        by_severity[severity] = db.query(Alert).filter(Alert.severity == severity).count()
    
    for status in ["OPEN", "ACKNOWLEDGED", "RESOLVED", "FALSE_POSITIVE"]:
        by_status[status] = db.query(Alert).filter(Alert.status == status).count()
    
    return {
        "total": total,
        "by_severity": by_severity,
        "by_status": by_status
    }
