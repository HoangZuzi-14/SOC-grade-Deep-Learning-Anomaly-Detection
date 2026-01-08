"""
Utility functions for testing
"""
import torch
import numpy as np
from typing import List


def create_mock_sequences(count: int = 10, length: int = 5, num_labels: int = 10) -> List[List[int]]:
    """Create mock log sequences for testing"""
    return [
        [np.random.randint(0, num_labels) for _ in range(length)]
        for _ in range(count)
    ]


def create_mock_scores(count: int = 10) -> List[float]:
    """Create mock anomaly scores for testing"""
    return [np.random.uniform(0, 10) for _ in range(count)]


def assert_model_output_valid(output: torch.Tensor, expected_shape: tuple):
    """Assert model output is valid"""
    assert output.shape == expected_shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    assert output.dtype == torch.float32
