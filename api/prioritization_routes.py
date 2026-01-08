"""
API routes for alert prioritization
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from datetime import datetime, timedelta

from api.database import get_db, Alert
from prioritization.priority_calculator import PriorityCalculator, AlertPrioritizer

router = APIRouter(prefix="/api/v1/prioritization", tags=["prioritization"])

# Global prioritizer
prioritizer = AlertPrioritizer()


@router.post("/calculate")
async def calculate_priority(
    alert_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Calculate priority for a specific alert"""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    # Get context (similar alerts, recent alerts)
    similar_count = db.query(Alert).filter(
        and_(
            Alert.sequence == alert.sequence,
            Alert.id != alert.id
        )
    ).count()
    
    recent_count = db.query(Alert).filter(
        and_(
            Alert.created_at >= datetime.utcnow() - timedelta(hours=1),
            Alert.id != alert.id
        )
    ).count()
    
    # Convert alert to dict
    alert_dict = {
        "id": alert.id,
        "score": alert.score,
        "severity": alert.severity,
        "timestamp": alert.created_at,
        "sequence": alert.sequence,
        "status": alert.status,
        "acknowledged_at": alert.acknowledged_at,
        "resolved_at": alert.resolved_at,
        "similar_alerts_count": similar_count,
        "recent_alerts_count": recent_count
    }
    
    # Calculate priority
    context = prioritizer._create_context(alert_dict, [alert_dict])
    priority_info = prioritizer.calculator.calculate_priority(context)
    
    # Update alert in database
    alert.priority_score = priority_info["priority_score"]
    alert.priority_rank = priority_info["priority_rank"]
    alert.urgency = priority_info["urgency"]
    db.commit()
    
    return {
        "alert_id": alert.id,
        **priority_info
    }


@router.post("/calculate-all")
async def calculate_all_priorities(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Calculate priorities for all open alerts"""
    alerts = db.query(Alert).filter(Alert.status == "OPEN").all()
    
    if not alerts:
        return {
            "message": "No open alerts to prioritize",
            "updated": 0
        }
    
    # Convert to dicts
    alert_dicts = [
        {
            "id": a.id,
            "score": a.score,
            "severity": a.severity,
            "timestamp": a.created_at,
            "sequence": a.sequence,
            "status": a.status,
            "acknowledged_at": a.acknowledged_at,
            "resolved_at": a.resolved_at
        }
        for a in alerts
    ]
    
    # Prioritize
    prioritized = prioritizer.prioritize_alerts(alert_dicts, include_context=True)
    
    # Update database
    updated = 0
    for alert_data in prioritized:
        alert = db.query(Alert).filter(Alert.id == alert_data["id"]).first()
        if alert:
            alert.priority_score = alert_data["priority_score"]
            alert.priority_rank = alert_data["priority_rank"]
            alert.urgency = alert_data["urgency"]
            updated += 1
    
    db.commit()
    
    return {
        "updated": updated,
        "total": len(alerts),
        "message": f"Updated priorities for {updated} alerts"
    }


@router.get("/top")
async def get_top_priority_alerts(
    limit: int = Query(default=10, ge=1, le=100),
    min_priority: Optional[str] = None,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get top priority alerts"""
    query = db.query(Alert).filter(Alert.status == "OPEN")
    
    if min_priority:
        priority_order = ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        if min_priority in priority_order:
            min_index = priority_order.index(min_priority)
            valid_ranks = priority_order[min_index:]
            query = query.filter(Alert.priority_rank.in_(valid_ranks))
    
    alerts = query.order_by(
        Alert.priority_score.desc(),
        Alert.created_at.desc()
    ).limit(limit).all()
    
    return {
        "alerts": [
            {
                "id": a.id,
                "score": a.score,
                "severity": a.severity,
                "priority_score": a.priority_score,
                "priority_rank": a.priority_rank,
                "urgency": a.urgency,
                "timestamp": a.created_at.isoformat(),
                "sequence": a.sequence,
                "status": a.status
            }
            for a in alerts
        ],
        "count": len(alerts)
    }


@router.get("/critical")
async def get_critical_alerts(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get critical priority alerts"""
    alerts = db.query(Alert).filter(
        and_(
            Alert.status == "OPEN",
            Alert.priority_rank == "CRITICAL"
        )
    ).order_by(Alert.priority_score.desc()).all()
    
    return {
        "alerts": [
            {
                "id": a.id,
                "score": a.score,
                "severity": a.severity,
                "priority_score": a.priority_score,
                "priority_rank": a.priority_rank,
                "urgency": a.urgency,
                "timestamp": a.created_at.isoformat(),
                "sequence": a.sequence
            }
            for a in alerts
        ],
        "count": len(alerts)
    }


@router.get("/high-urgency")
async def get_high_urgency_alerts(
    urgency_threshold: float = Query(default=0.7, ge=0.0, le=1.0),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get alerts with high urgency"""
    alerts = db.query(Alert).filter(
        and_(
            Alert.status == "OPEN",
            Alert.urgency >= urgency_threshold
        )
    ).order_by(Alert.urgency.desc(), Alert.priority_score.desc()).all()
    
    return {
        "alerts": [
            {
                "id": a.id,
                "score": a.score,
                "severity": a.severity,
                "priority_score": a.priority_score,
                "priority_rank": a.priority_rank,
                "urgency": a.urgency,
                "timestamp": a.created_at.isoformat(),
                "sequence": a.sequence
            }
            for a in alerts
        ],
        "count": len(alerts),
        "threshold": urgency_threshold
    }


@router.get("/statistics")
async def get_prioritization_statistics(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get prioritization statistics"""
    total = db.query(Alert).filter(Alert.status == "OPEN").count()
    
    # Count by priority rank
    rank_counts = db.query(
        Alert.priority_rank,
        func.count(Alert.id)
    ).filter(Alert.status == "OPEN").group_by(Alert.priority_rank).all()
    
    rank_stats = {rank: count for rank, count in rank_counts}
    
    # Average priority score
    avg_score = db.query(func.avg(Alert.priority_score)).filter(
        Alert.status == "OPEN"
    ).scalar() or 0.0
    
    # Average urgency
    avg_urgency = db.query(func.avg(Alert.urgency)).filter(
        Alert.status == "OPEN"
    ).scalar() or 0.0
    
    return {
        "total_open_alerts": total,
        "by_priority_rank": rank_stats,
        "average_priority_score": round(avg_score, 2),
        "average_urgency": round(avg_urgency, 3)
    }
