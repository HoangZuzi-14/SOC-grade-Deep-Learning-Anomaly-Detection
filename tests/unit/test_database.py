"""
Unit tests for database models and functions
"""
import pytest
from datetime import datetime
from api.database import (
    LogEntry,
    Sequence,
    AnomalyScore,
    Alert,
    create_alert_from_score,
    get_recent_alerts,
    update_alert_status,
    get_alert_statistics
)


@pytest.mark.unit
@pytest.mark.database
class TestDatabaseModels:
    """Test database models"""
    
    def test_log_entry_creation(self, db_session):
        """Test creating a log entry"""
        log_entry = LogEntry(
            template_id=1,
            template_text="User <*> logged in",
            raw_log="User admin logged in",
            source="auth.log"
        )
        db_session.add(log_entry)
        db_session.commit()
        
        assert log_entry.id is not None
        assert log_entry.template_id == 1
        assert log_entry.source == "auth.log"
    
    def test_sequence_creation(self, db_session):
        """Test creating a sequence"""
        sequence = Sequence(
            sequence=[1, 2, 3, 4, 5],
            window_size=5
        )
        db_session.add(sequence)
        db_session.commit()
        
        assert sequence.id is not None
        assert len(sequence.sequence) == 5
    
    def test_anomaly_score_creation(self, db_session):
        """Test creating an anomaly score"""
        seq = Sequence(sequence=[1, 2, 3], window_size=3)
        db_session.add(seq)
        db_session.commit()
        
        score = AnomalyScore(
            sequence_id=seq.id,
            score=5.5,
            severity="HIGH",
            model_type="lstm"
        )
        db_session.add(score)
        db_session.commit()
        
        assert score.id is not None
        assert score.score == 5.5
        assert score.severity == "HIGH"
    
    def test_alert_creation(self, db_session):
        """Test creating an alert"""
        alert = Alert(
            sequence=[1, 2, 3, 4, 5],
            score=8.5,
            severity="HIGH",
            status="OPEN"
        )
        db_session.add(alert)
        db_session.commit()
        
        assert alert.id is not None
        assert alert.status == "OPEN"
        assert alert.priority > 0


@pytest.mark.unit
@pytest.mark.database
class TestDatabaseFunctions:
    """Test database utility functions"""
    
    def test_create_alert_from_score(self, db_session):
        """Test creating alert from score"""
        alert = create_alert_from_score(
            db=db_session,
            sequence=[1, 2, 3],
            score=9.5,
            severity="HIGH"
        )
        
        assert alert.id is not None
        assert alert.score == 9.5
        assert alert.severity == "HIGH"
        assert alert.status == "OPEN"
        assert alert.priority > 0
    
    def test_get_recent_alerts(self, db_session):
        """Test getting recent alerts"""
        # Create some alerts
        for i in range(5):
            alert = Alert(
                sequence=[i, i+1, i+2],
                score=float(i),
                severity="HIGH" if i > 2 else "LOW",
                status="OPEN"
            )
            db_session.add(alert)
        db_session.commit()
        
        alerts = get_recent_alerts(db_session, limit=3)
        assert len(alerts) == 3
    
    def test_update_alert_status(self, db_session):
        """Test updating alert status"""
        alert = Alert(
            sequence=[1, 2, 3],
            score=5.0,
            severity="MED",
            status="OPEN"
        )
        db_session.add(alert)
        db_session.commit()
        
        updated = update_alert_status(
            db_session,
            alert.id,
            "ACKNOWLEDGED",
            user="test_user"
        )
        
        assert updated.status == "ACKNOWLEDGED"
        assert updated.acknowledged_by == "test_user"
        assert updated.acknowledged_at is not None
    
    def test_get_alert_statistics(self, db_session):
        """Test getting alert statistics"""
        # Create alerts with different severities
        for severity in ["HIGH", "MED", "LOW", "HIGH"]:
            alert = Alert(
                sequence=[1, 2, 3],
                score=5.0,
                severity=severity,
                status="OPEN"
            )
            db_session.add(alert)
        db_session.commit()
        
        stats = get_alert_statistics(db_session)
        
        assert stats["total"] == 4
        assert stats["by_severity"]["HIGH"] == 2
        assert stats["by_severity"]["MED"] == 1
        assert stats["by_severity"]["LOW"] == 1
