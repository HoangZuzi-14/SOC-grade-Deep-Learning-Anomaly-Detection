"""
Unit tests for scoring module
"""
import pytest
import math
from scoring.anomaly_score import (
    compute_thresholds,
    severity_from_score,
    alert_from_severity,
    score_to_soc_decision,
    Thresholds
)


@pytest.mark.unit
class TestThresholds:
    """Test threshold computation"""
    
    def test_compute_thresholds_basic(self, sample_scores):
        """Test basic threshold computation"""
        thresholds = compute_thresholds(sample_scores)
        
        assert isinstance(thresholds, Thresholds)
        assert thresholds.p95 > thresholds.p99
        assert thresholds.p99 > thresholds.p999
        assert thresholds.p95 >= 0
        assert thresholds.p99 >= 0
        assert thresholds.p999 >= 0
    
    def test_compute_thresholds_empty(self):
        """Test with empty scores"""
        thresholds = compute_thresholds([])
        assert thresholds.p95 == 0.0
        assert thresholds.p99 == 0.0
        assert thresholds.p999 == 0.0
    
    def test_compute_thresholds_single_value(self):
        """Test with single score"""
        thresholds = compute_thresholds([5.0])
        assert thresholds.p95 == 5.0
        assert thresholds.p99 == 5.0
        assert thresholds.p999 == 5.0


@pytest.mark.unit
class TestSeverityClassification:
    """Test severity classification"""
    
    def test_severity_from_score(self):
        """Test severity classification from score"""
        thresholds = Thresholds(p95=5.0, p99=8.0, p999=10.0)
        
        assert severity_from_score(3.0, thresholds) == "NONE"
        assert severity_from_score(6.0, thresholds) == "LOW"
        assert severity_from_score(9.0, thresholds) == "MED"
        assert severity_from_score(11.0, thresholds) == "HIGH"
    
    def test_alert_from_severity(self):
        """Test alert generation from severity"""
        assert alert_from_severity("HIGH", "MED") == True
        assert alert_from_severity("MED", "MED") == True
        assert alert_from_severity("LOW", "MED") == False
        assert alert_from_severity("NONE", "MED") == False
    
    def test_score_to_soc_decision(self):
        """Test complete SOC decision from score"""
        thresholds = Thresholds(p95=5.0, p99=8.0, p999=10.0)
        
        alert, severity = score_to_soc_decision(11.0, thresholds, min_alert="MED")
        assert alert == True
        assert severity == "HIGH"
        
        alert, severity = score_to_soc_decision(3.0, thresholds, min_alert="MED")
        assert alert == False
        assert severity == "NONE"
