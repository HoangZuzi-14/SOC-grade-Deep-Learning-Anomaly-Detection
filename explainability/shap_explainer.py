"""
SHAP-based model explainability for anomaly detection
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import shap
from pathlib import Path
import json


class DeepLogSHAPExplainer:
    """
    SHAP explainer for DeepLog LSTM model
    """
    
    def __init__(self, model, background_data: Optional[List[List[int]]] = None):
        """
        Args:
            model: Trained DeepLogLSTM model
            background_data: Background sequences for SHAP (optional)
        """
        self.model = model
        self.model.eval()
        self.background_data = background_data or []
        self.explainer = None
        
    def prepare_background(self, sequences: List[List[int]], max_samples: int = 100):
        """Prepare background data for SHAP"""
        if len(sequences) > max_samples:
            # Sample random sequences
            indices = np.random.choice(len(sequences), max_samples, replace=False)
            self.background_data = [sequences[i] for i in indices]
        else:
            self.background_data = sequences
    
    def create_explainer(self, background_sequences: Optional[List[List[int]]] = None):
        """Create SHAP explainer with background data"""
        if background_sequences:
            self.prepare_background(background_sequences)
        
        # Use DeepExplainer for neural networks
        # For LSTM, we'll use a wrapper function
        def model_wrapper(sequences):
            """Wrapper function for SHAP"""
            self.model.eval()
            scores = []
            
            with torch.no_grad():
                for seq in sequences:
                    if len(seq) == 0:
                        scores.append(0.0)
                        continue
                    
                    # Prepare input
                    window_size = len(seq)
                    if window_size > 5:
                        seq = seq[-5:]
                    elif window_size < 5:
                        pad_len = 5 - window_size
                        seq = ([0] * pad_len) + seq
                    
                    x = torch.tensor(seq[:-1], dtype=torch.long).unsqueeze(0)
                    y = torch.tensor(seq[-1], dtype=torch.long)
                    
                    # Get prediction
                    logits = self.model(x).squeeze(0)
                    logp = F.log_softmax(logits, dim=-1)
                    score = float((-logp[y]).item())
                    
                    scores.append(score)
            
            return np.array(scores)
        
        # Create background tensor
        if self.background_data:
            background_tensor = self._sequences_to_tensor(self.background_data[:50])
        else:
            # Create dummy background
            background_tensor = torch.zeros((10, 4), dtype=torch.long)
        
        # Use KernelExplainer for flexibility
        self.explainer = shap.KernelExplainer(
            model_wrapper,
            background_tensor.numpy()
        )
        
        return self.explainer
    
    def _sequences_to_tensor(self, sequences: List[List[int]], window_size: int = 5) -> torch.Tensor:
        """Convert sequences to tensor"""
        tensors = []
        for seq in sequences:
            if len(seq) > window_size:
                seq = seq[-window_size:]
            elif len(seq) < window_size:
                pad_len = window_size - len(seq)
                seq = ([0] * pad_len) + seq
            
            # Use last window_size-1 for input
            tensors.append(seq[:-1])
        
        # Pad to same length
        max_len = max(len(t) for t in tensors) if tensors else window_size - 1
        padded = []
        for t in tensors:
            if len(t) < max_len:
                t = ([0] * (max_len - len(t))) + t
            padded.append(t[:max_len])
        
        return torch.tensor(padded, dtype=torch.long)
    
    def explain_sequence(
        self,
        sequence: List[int],
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Explain a single sequence using SHAP
        
        Args:
            sequence: Input sequence
            num_samples: Number of samples for SHAP
            
        Returns:
            Explanation dictionary with SHAP values
        """
        if self.explainer is None:
            self.create_explainer()
        
        # Prepare sequence
        window_size = 5  # Default, should match model
        if len(sequence) > window_size:
            seq = sequence[-window_size:]
        elif len(sequence) < window_size:
            pad_len = window_size - len(sequence)
            seq = ([0] * pad_len) + sequence
        else:
            seq = sequence
        
        # Prepare input for SHAP (last window_size-1 tokens)
        input_seq = seq[:-1]
        input_tensor = torch.tensor([input_seq], dtype=torch.long)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(
            input_tensor.numpy(),
            nsamples=num_samples
        )
        
        # Get prediction
        with torch.no_grad():
            x = torch.tensor([input_seq], dtype=torch.long)
            y = torch.tensor(seq[-1], dtype=torch.long)
            logits = self.model(x).squeeze(0)
            logp = F.log_softmax(logits, dim=-1)
            score = float((-logp[y]).item())
            probs = F.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=min(5, logits.numel()))
        
        # Format SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        shap_array = np.array(shap_values).flatten()
        
        # Get feature importance (contribution of each position)
        feature_importance = []
        for i, val in enumerate(shap_array):
            if i < len(input_seq):
                feature_importance.append({
                    "position": i,
                    "template_id": input_seq[i],
                    "shap_value": float(val),
                    "contribution": float(val)
                })
        
        # Sort by absolute contribution
        feature_importance.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        
        return {
            "sequence": sequence,
            "input_sequence": input_seq,
            "target": int(seq[-1]),
            "score": score,
            "shap_values": shap_array.tolist(),
            "feature_importance": feature_importance,
            "top_predictions": [
                {"template_id": int(idx), "probability": float(prob)}
                for idx, prob in zip(top_indices, top_probs)
            ],
            "base_value": float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0.0
        }
    
    def explain_batch(
        self,
        sequences: List[List[int]],
        num_samples: int = 100
    ) -> List[Dict[str, Any]]:
        """Explain multiple sequences"""
        explanations = []
        for seq in sequences:
            try:
                explanation = self.explain_sequence(seq, num_samples)
                explanations.append(explanation)
            except Exception as e:
                explanations.append({
                    "sequence": seq,
                    "error": str(e)
                })
        return explanations
    
    def get_summary_plot_data(self, sequences: List[List[int]], max_samples: int = 50):
        """Get data for SHAP summary plot"""
        if len(sequences) > max_samples:
            indices = np.random.choice(len(sequences), max_samples, replace=False)
            sequences = [sequences[i] for i in indices]
        
        explanations = self.explain_batch(sequences, num_samples=50)
        
        # Aggregate SHAP values
        all_shap_values = []
        all_features = []
        
        for exp in explanations:
            if "error" not in exp:
                all_shap_values.append(exp["shap_values"])
                all_features.append(exp["input_sequence"])
        
        return {
            "shap_values": all_shap_values,
            "features": all_features,
            "explanations": explanations
        }


class AttentionExplainer:
    """
    Simple attention-based explanation (alternative to SHAP)
    Uses gradient-based attention
    """
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def explain_sequence(self, sequence: List[int]) -> Dict[str, Any]:
        """Explain using gradient-based attention"""
        window_size = 5
        if len(sequence) > window_size:
            seq = sequence[-window_size:]
        elif len(sequence) < window_size:
            pad_len = window_size - len(sequence)
            seq = ([0] * pad_len) + sequence
        else:
            seq = sequence
        
        input_seq = seq[:-1]
        x = torch.tensor([input_seq], dtype=torch.long, requires_grad=True)
        y = torch.tensor(seq[-1], dtype=torch.long)
        
        # Forward pass
        logits = self.model(x).squeeze(0)
        loss = F.cross_entropy(logits.unsqueeze(0), y.unsqueeze(0))
        
        # Backward pass
        loss.backward()
        
        # Get gradients as attention
        if x.grad is not None:
            attention = x.grad.abs().squeeze(0).numpy()
        else:
            attention = np.ones(len(input_seq))
        
        # Normalize
        attention = attention / (attention.sum() + 1e-8)
        
        feature_importance = []
        for i, (tid, attn) in enumerate(zip(input_seq, attention)):
            feature_importance.append({
                "position": i,
                "template_id": int(tid),
                "attention": float(attn),
                "contribution": float(attn)
            })
        
        feature_importance.sort(key=lambda x: x["contribution"], reverse=True)
        
        return {
            "sequence": sequence,
            "input_sequence": input_seq,
            "target": int(seq[-1]),
            "attention_weights": attention.tolist(),
            "feature_importance": feature_importance,
            "method": "gradient_attention"
        }
