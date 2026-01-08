"""
Alert prioritization logic
Calculates priority scores based on multiple factors
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import math


@dataclass
class AlertContext:
    """Context information for alert prioritization"""
    alert_id: int
    score: float
    severity: str
    timestamp: datetime
    sequence: List[int]
    status: str = "OPEN"
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    similar_alerts_count: int = 0
    recent_alerts_count: int = 0
    source: Optional[str] = None
    user: Optional[str] = None
    ip_address: Optional[str] = None


class PriorityCalculator:
    """
    Calculate alert priority based on multiple factors
    """
    
    # Severity weights
    SEVERITY_WEIGHTS = {
        "HIGH": 1.0,
        "MED": 0.6,
        "LOW": 0.3,
        "NONE": 0.1
    }
    
    # Score thresholds
    SCORE_THRESHOLDS = {
        "critical": 10.0,
        "high": 7.0,
        "medium": 4.0,
        "low": 2.0
    }
    
    def __init__(
        self,
        severity_weight: float = 0.4,
        score_weight: float = 0.3,
        recency_weight: float = 0.15,
        frequency_weight: float = 0.15
    ):
        """
        Args:
            severity_weight: Weight for severity factor (0-1)
            score_weight: Weight for score factor (0-1)
            recency_weight: Weight for recency factor (0-1)
            frequency_weight: Weight for frequency factor (0-1)
        """
        total_weight = severity_weight + score_weight + recency_weight + frequency_weight
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        self.severity_weight = severity_weight
        self.score_weight = score_weight
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
    
    def calculate_priority(self, context: AlertContext) -> Dict[str, Any]:
        """
        Calculate priority score and rank
        
        Returns:
            Dictionary with priority_score, priority_rank, and factors
        """
        # Calculate individual factors
        severity_factor = self._severity_factor(context.severity)
        score_factor = self._score_factor(context.score)
        recency_factor = self._recency_factor(context.timestamp)
        frequency_factor = self._frequency_factor(
            context.similar_alerts_count,
            context.recent_alerts_count
        )
        
        # Calculate weighted priority score (0-100)
        priority_score = (
            severity_factor * self.severity_weight * 100 +
            score_factor * self.score_weight * 100 +
            recency_factor * self.recency_weight * 100 +
            frequency_factor * self.frequency_weight * 100
        )
        
        # Determine priority rank
        priority_rank = self._score_to_rank(priority_score)
        
        # Calculate urgency (0-1)
        urgency = self._calculate_urgency(context)
        
        return {
            "priority_score": round(priority_score, 2),
            "priority_rank": priority_rank,
            "urgency": round(urgency, 3),
            "factors": {
                "severity": {
                    "value": severity_factor,
                    "weight": self.severity_weight,
                    "contribution": severity_factor * self.severity_weight * 100
                },
                "score": {
                    "value": score_factor,
                    "weight": self.score_weight,
                    "contribution": score_factor * self.score_weight * 100
                },
                "recency": {
                    "value": recency_factor,
                    "weight": self.recency_weight,
                    "contribution": recency_factor * self.recency_weight * 100
                },
                "frequency": {
                    "value": frequency_factor,
                    "weight": self.frequency_weight,
                    "contribution": frequency_factor * self.frequency_weight * 100
                }
            }
        }
    
    def _severity_factor(self, severity: str) -> float:
        """Calculate severity factor (0-1)"""
        return self.SEVERITY_WEIGHTS.get(severity.upper(), 0.1)
    
    def _score_factor(self, score: float) -> float:
        """Calculate score factor (0-1) using normalized score"""
        # Normalize score to 0-1 range
        # Assuming scores typically range from 0-20
        normalized = min(score / 20.0, 1.0)
        
        # Apply sigmoid for better distribution
        sigmoid = 1 / (1 + math.exp(-5 * (normalized - 0.5)))
        
        return sigmoid
    
    def _recency_factor(self, timestamp: datetime) -> float:
        """Calculate recency factor (0-1), newer = higher"""
        now = datetime.utcnow()
        age_seconds = (now - timestamp).total_seconds()
        
        # Decay over 24 hours
        decay_hours = 24
        decay_seconds = decay_hours * 3600
        
        # Exponential decay
        factor = math.exp(-age_seconds / decay_seconds)
        
        return max(0.0, min(1.0, factor))
    
    def _frequency_factor(
        self,
        similar_count: int,
        recent_count: int
    ) -> float:
        """Calculate frequency factor (0-1), more frequent = higher"""
        # Combine similar and recent alerts
        total_count = similar_count + recent_count
        
        # Logarithmic scaling to prevent extreme values
        if total_count == 0:
            return 0.0
        
        # Scale: 1 alert = 0.2, 5 alerts = 0.6, 10+ alerts = 1.0
        factor = min(1.0, 0.2 + (total_count / 10.0) * 0.8)
        
        return factor
    
    def _score_to_rank(self, score: float) -> str:
        """Convert priority score to rank"""
        if score >= 80:
            return "CRITICAL"
        elif score >= 60:
            return "HIGH"
        elif score >= 40:
            return "MEDIUM"
        elif score >= 20:
            return "LOW"
        else:
            return "INFO"
    
    def _calculate_urgency(self, context: AlertContext) -> float:
        """
        Calculate urgency (0-1) based on multiple factors
        Higher urgency = needs immediate attention
        """
        # Base urgency from severity
        severity_urgency = {
            "HIGH": 0.9,
            "MED": 0.6,
            "LOW": 0.3,
            "NONE": 0.1
        }.get(context.severity.upper(), 0.1)
        
        # Score urgency (normalized)
        score_urgency = min(context.score / 15.0, 1.0)
        
        # Recency urgency (newer = more urgent)
        now = datetime.utcnow()
        age_hours = (now - context.timestamp).total_seconds() / 3600
        recency_urgency = max(0.0, 1.0 - (age_hours / 12.0))
        
        # Frequency urgency (more alerts = more urgent)
        frequency_urgency = min((context.similar_alerts_count + context.recent_alerts_count) / 5.0, 1.0)
        
        # Status urgency (unacknowledged = more urgent)
        status_urgency = 1.0 if context.status == "OPEN" else 0.3
        
        # Weighted average
        urgency = (
            severity_urgency * 0.3 +
            score_urgency * 0.25 +
            recency_urgency * 0.2 +
            frequency_urgency * 0.15 +
            status_urgency * 0.1
        )
        
        return min(1.0, max(0.0, urgency))


class AlertPrioritizer:
    """
    High-level interface for alert prioritization
    """
    
    def __init__(self, calculator: Optional[PriorityCalculator] = None):
        self.calculator = calculator or PriorityCalculator()
    
    def prioritize_alerts(
        self,
        alerts: List[Dict[str, Any]],
        include_context: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Prioritize a list of alerts
        
        Args:
            alerts: List of alert dictionaries
            include_context: Whether to include context information
            
        Returns:
            Sorted list of alerts with priority information
        """
        prioritized = []
        
        for alert in alerts:
            # Create context
            context = self._create_context(alert, alerts if include_context else [])
            
            # Calculate priority
            priority_info = self.calculator.calculate_priority(context)
            
            # Add priority info to alert
            prioritized_alert = alert.copy()
            prioritized_alert.update(priority_info)
            prioritized_alert["context"] = context.__dict__ if include_context else None
            
            prioritized.append(prioritized_alert)
        
        # Sort by priority score (descending)
        prioritized.sort(key=lambda x: x["priority_score"], reverse=True)
        
        return prioritized
    
    def _create_context(
        self,
        alert: Dict[str, Any],
        all_alerts: List[Dict[str, Any]]
    ) -> AlertContext:
        """Create AlertContext from alert data"""
        # Parse timestamp
        timestamp = alert.get("timestamp") or alert.get("created_at")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        elif not isinstance(timestamp, datetime):
            timestamp = datetime.utcnow()
        
        # Count similar alerts (same sequence pattern)
        sequence = alert.get("sequence", [])
        similar_count = sum(
            1 for a in all_alerts
            if a.get("id") != alert.get("id") and
            a.get("sequence") == sequence
        )
        
        # Count recent alerts (last hour)
        now = datetime.utcnow()
        recent_count = sum(
            1 for a in all_alerts
            if a.get("id") != alert.get("id")
        )
        # Filter by timestamp if available
        if all_alerts:
            recent_count = sum(
                1 for a in all_alerts
                if a.get("id") != alert.get("id")
            )
        
        return AlertContext(
            alert_id=alert.get("id", 0),
            score=alert.get("score", 0.0),
            severity=alert.get("severity", "NONE"),
            timestamp=timestamp,
            sequence=sequence,
            status=alert.get("status", "OPEN"),
            acknowledged_at=alert.get("acknowledged_at"),
            resolved_at=alert.get("resolved_at"),
            similar_alerts_count=similar_count,
            recent_alerts_count=recent_count,
            source=alert.get("source"),
            user=alert.get("user"),
            ip_address=alert.get("ip_address")
        )
    
    def get_top_priority_alerts(
        self,
        alerts: List[Dict[str, Any]],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top N priority alerts"""
        prioritized = self.prioritize_alerts(alerts)
        return prioritized[:limit]
    
    def get_critical_alerts(
        self,
        alerts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get only critical priority alerts"""
        prioritized = self.prioritize_alerts(alerts)
        return [a for a in prioritized if a["priority_rank"] == "CRITICAL"]
    
    def get_high_urgency_alerts(
        self,
        alerts: List[Dict[str, Any]],
        urgency_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Get alerts with urgency above threshold"""
        prioritized = self.prioritize_alerts(alerts)
        return [a for a in prioritized if a["urgency"] >= urgency_threshold]
