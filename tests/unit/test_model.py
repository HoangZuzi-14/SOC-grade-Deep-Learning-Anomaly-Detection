"""
Unit tests for LSTM model
"""
import pytest
import torch
from model.deeplog_lstm import DeepLogLSTM


@pytest.mark.unit
@pytest.mark.model
class TestDeepLogLSTM:
    """Test DeepLog LSTM model"""
    
    def test_model_initialization(self):
        """Test model can be initialized"""
        model = DeepLogLSTM(
            num_labels=10,
            embedding_dim=16,
            hidden_size=64,
            num_layers=1
        )
        
        assert model is not None
        assert model.embedding.num_embeddings == 10
        assert model.embedding.embedding_dim == 16
    
    def test_model_forward(self, sample_model):
        """Test model forward pass"""
        batch_size = 2
        window_size = 5
        
        # Create input tensor
        x = torch.randint(0, 10, (batch_size, window_size))
        
        # Forward pass
        output = sample_model(x)
        
        assert output.shape == (batch_size, 10)  # (batch, num_labels)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_model_parameters(self, sample_model):
        """Test model has trainable parameters"""
        total_params = sum(p.numel() for p in sample_model.parameters())
        trainable_params = sum(p.numel() for p in sample_model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert total_params == trainable_params  # All params should be trainable
    
    def test_model_different_configs(self):
        """Test model with different configurations"""
        configs = [
            {"num_labels": 5, "embedding_dim": 8, "hidden_size": 32, "num_layers": 1},
            {"num_labels": 20, "embedding_dim": 32, "hidden_size": 128, "num_layers": 2},
        ]
        
        for config in configs:
            model = DeepLogLSTM(**config)
            x = torch.randint(0, config["num_labels"], (1, 5))
            output = model(x)
            assert output.shape == (1, config["num_labels"])
