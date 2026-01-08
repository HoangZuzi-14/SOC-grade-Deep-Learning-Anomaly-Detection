"""
Integration tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.database import Base, engine, SessionLocal, init_db


@pytest.fixture(scope="module")
def test_client():
    """Create test client"""
    # Use in-memory database for testing
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    
    def override_get_db():
        try:
            db = TestSessionLocal()
            yield db
        finally:
            db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    
    client = TestClient(app)
    yield client
    
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()


@pytest.mark.integration
@pytest.mark.api
class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self, test_client):
        """Test root health check"""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"
    
    def test_health_check_structure(self, test_client):
        """Test health check response structure"""
        response = test_client.get("/")
        data = response.json()
        
        assert "model_loaded" in data
        assert "config" in data
        assert "database_connected" in data


@pytest.mark.integration
@pytest.mark.api
class TestScoreEndpoint:
    """Test scoring endpoints"""
    
    def test_score_single_sequence(self, test_client):
        """Test scoring a single sequence"""
        response = test_client.post(
            "/api/v1/score",
            json={"sequence": [1, 2, 3, 4, 5], "model_type": "lstm"}
        )
        
        # May fail if model not loaded, but should return proper error
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "score" in data
            assert "severity" in data
            assert "alert" in data
            assert "sequence" in data
    
    def test_score_batch(self, test_client):
        """Test batch scoring"""
        response = test_client.post(
            "/api/v1/score/batch",
            json={
                "sequences": [[1, 2, 3], [4, 5, 6]],
                "model_type": "lstm"
            }
        )
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            assert "count" in data
            assert data["count"] == 2


@pytest.mark.integration
@pytest.mark.api
class TestAlertEndpoints:
    """Test alert endpoints"""
    
    def test_create_alert(self, test_client):
        """Test creating an alert"""
        response = test_client.post(
            "/api/v1/alerts",
            json={
                "sequence": [1, 2, 3, 4, 5],
                "score": 8.5,
                "severity": "HIGH"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["severity"] == "HIGH"
        assert data["score"] == 8.5
    
    def test_list_alerts(self, test_client):
        """Test listing alerts"""
        # Create an alert first
        test_client.post(
            "/api/v1/alerts",
            json={
                "sequence": [1, 2, 3],
                "score": 5.0,
                "severity": "MED"
            }
        )
        
        response = test_client.get("/api/v1/alerts?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_alert_by_id(self, test_client):
        """Test getting alert by ID"""
        # Create an alert
        create_response = test_client.post(
            "/api/v1/alerts",
            json={
                "sequence": [1, 2, 3],
                "score": 6.0,
                "severity": "MED"
            }
        )
        alert_id = create_response.json()["id"]
        
        # Get the alert
        response = test_client.get(f"/api/v1/alerts/{alert_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == alert_id
    
    def test_update_alert_status(self, test_client):
        """Test updating alert status"""
        # Create an alert
        create_response = test_client.post(
            "/api/v1/alerts",
            json={
                "sequence": [1, 2, 3],
                "score": 7.0,
                "severity": "HIGH"
            }
        )
        alert_id = create_response.json()["id"]
        
        # Update status
        response = test_client.patch(
            f"/api/v1/alerts/{alert_id}",
            json={"status": "ACKNOWLEDGED", "user": "test_user"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ACKNOWLEDGED"
    
    def test_alert_statistics(self, test_client):
        """Test alert statistics endpoint"""
        # Create some alerts
        for severity in ["HIGH", "MED", "LOW"]:
            test_client.post(
                "/api/v1/alerts",
                json={
                    "sequence": [1, 2, 3],
                    "score": 5.0,
                    "severity": severity
                }
            )
        
        response = test_client.get("/api/v1/alerts/statistics/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "by_severity" in data
        assert "by_status" in data


@pytest.mark.integration
@pytest.mark.api
class TestModelInfoEndpoint:
    """Test model info endpoint"""
    
    def test_model_info(self, test_client):
        """Test model info endpoint"""
        response = test_client.get("/api/v1/model/info")
        
        # May return 503 if model not loaded
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "model_loaded" in data
            assert "config" in data
