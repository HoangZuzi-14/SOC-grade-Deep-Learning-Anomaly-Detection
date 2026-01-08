"""
Pytest configuration and fixtures
"""
import pytest
import sys
from pathlib import Path
from typing import Generator
import torch
import tempfile
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.database import Base, engine, SessionLocal, get_db
from model.deeplog_lstm import DeepLogLSTM


@pytest.fixture(scope="session")
def test_db():
    """Create a temporary test database"""
    # Use in-memory SQLite for testing
    test_db_path = ":memory:"
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    test_engine = create_engine(f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    
    yield TestSessionLocal()
    
    Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()


@pytest.fixture
def db_session(test_db):
    """Provide a database session for each test"""
    yield test_db
    test_db.rollback()


@pytest.fixture
def sample_model():
    """Create a sample LSTM model for testing"""
    num_labels = 10
    model = DeepLogLSTM(
        num_labels=num_labels,
        embedding_dim=16,
        hidden_size=64,
        num_layers=1
    )
    model.eval()
    return model


@pytest.fixture
def sample_sequences():
    """Sample log sequences for testing"""
    return [
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [1, 1, 2, 2, 3],
        [5, 4, 3, 2, 1],
    ]


@pytest.fixture
def sample_scores():
    """Sample anomaly scores for testing"""
    return [0.5, 1.2, 3.5, 8.9, 2.1, 0.3, 5.6, 1.8]


@pytest.fixture
def mock_model_path(tmp_path):
    """Create a mock model checkpoint file"""
    model_path = tmp_path / "test_model.pth"
    
    # Create a minimal model checkpoint
    num_labels = 10
    model = DeepLogLSTM(
        num_labels=num_labels,
        embedding_dim=16,
        hidden_size=64,
        num_layers=1
    )
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_labels": num_labels,
        "window_size": 5,
        "embedding_dim": 16,
        "hidden_size": 64,
        "num_layers": 1,
    }
    
    torch.save(checkpoint, model_path)
    return str(model_path)


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "sequences").mkdir()
    (data_dir / "parsed").mkdir()
    (data_dir / "results").mkdir()
    return str(data_dir)
