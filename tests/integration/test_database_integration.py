"""
Integration tests for database operations
"""
import pytest
from api.database import (
    SessionLocal,
    LogEntry,
    Sequence,
    AnomalyScore,
    Alert,
    create_alert_from_score,
    get_recent_alerts,
    update_alert_status,
    get_alert_statistics
)


@pytest.mark.integration
@pytest.mark.database
class TestDatabaseIntegration:
    """Integration tests for database operations"""
    
    def test_full_alert_workflow(self, db_session):
        """Test complete alert workflow"""
        # 1. Create log entry
        log_entry = LogEntry(
            template_id=1,
            template_text="User <*> logged in",
            source="auth.log"
        )
        db_session.add(log_entry)
        db_session.commit()
        
        # 2. Create sequence
        sequence = Sequence(
            sequence=[1, 2, 3, 4, 5],
            window_size=5
        )
        db_session.add(sequence)
        db_session.commit()
        
        # 3. Create anomaly score
        score = AnomalyScore(
            sequence_id=sequence.id,
            score=9.5,
            severity="HIGH",
            model_type="lstm"
        )
        db_session.add(score)
        db_session.commit()
        
        # 4. Create alert
        alert = create_alert_from_score(
            db=db_session,
            sequence=sequence.sequence,
            score=score.score,
            severity=score.severity,
            score_id=score.id,
            sequence_id=sequence.id
        )
        
        assert alert.id is not None
        assert alert.score_id == score.id
        assert alert.sequence_id == sequence.id
        
        # 5. Get recent alerts
        alerts = get_recent_alerts(db_session, limit=10)
        assert len(alerts) >= 1
        assert alerts[0].id == alert.id
        
        # 6. Update alert status
        updated = update_alert_status(
            db_session,
            alert.id,
            "RESOLVED",
            user="admin"
        )
        assert updated.status == "RESOLVED"
        assert updated.resolved_at is not None
        
        # 7. Get statistics
        stats = get_alert_statistics(db_session)
        assert stats["total"] >= 1
        assert stats["by_severity"]["HIGH"] >= 1
