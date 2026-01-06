"""
Deep Learning Model Analysis Tools
Provides utilities for analyzing LSTM model architecture, parameters, and embeddings.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional

# Import DeepLogLSTM - handle both relative and absolute imports
try:
    from .deeplog_lstm import DeepLogLSTM
except ImportError:
    try:
        from model.deeplog_lstm import DeepLogLSTM
    except ImportError:
        from deeplog_lstm import DeepLogLSTM


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters in the model.
    
    Returns:
        Dictionary with 'total', 'trainable', and 'non_trainable' counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': non_trainable
    }


def print_model_summary(model: nn.Module, input_size: Tuple[int, ...] = (64, 5)):
    """
    Print detailed model architecture summary.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch_size, window_size)
    """
    print("=" * 80)
    print("DEEP LEARNING MODEL SUMMARY")
    print("=" * 80)
    
    # Model architecture
    print("\n[ARCHITECTURE]")
    print(model)
    
    # Parameter count
    params = count_parameters(model)
    print("\n[PARAMETERS]")
    print(f"  Total parameters      : {params['total']:,}")
    print(f"  Trainable parameters : {params['trainable']:,}")
    print(f"  Non-trainable         : {params['non_trainable']:,}")
    
    # Model size estimation
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    print(f"\n[MODEL SIZE]")
    print(f"  Estimated size       : {model_size_mb:.2f} MB")
    
    # Layer-wise breakdown
    print("\n[LAYER BREAKDOWN]")
    for name, module in model.named_children():
        if isinstance(module, (nn.Embedding, nn.LSTM, nn.Linear)):
            layer_params = sum(p.numel() for p in module.parameters())
            print(f"  {name:20s} : {layer_params:,} parameters")
    
    # Forward pass test
    print("\n[FORWARD PASS TEST]")
    try:
        model.eval()
        with torch.no_grad():
            test_input = torch.randint(0, 100, input_size)
            output = model(test_input)
            print(f"  Input shape          : {test_input.shape}")
            print(f"  Output shape         : {output.shape}")
            print(f"  Output dtype         : {output.dtype}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("=" * 80)


def visualize_embeddings(model: DeepLogLSTM, num_labels: int, save_path: Optional[str] = None, 
                         method: str = 'tsne', top_n: int = 50):
    """
    Visualize learned embeddings using dimensionality reduction.
    
    Args:
        model: Trained DeepLogLSTM model
        num_labels: Number of unique log templates
        save_path: Path to save visualization
        method: 'tsne' or 'pca'
        top_n: Visualize top N most frequent templates
    """
    model.eval()
    embeddings = model.embedding.weight.data.cpu().numpy()
    
    if top_n < num_labels:
        # Use top N most frequent (simplified: use first N)
        embeddings = embeddings[:top_n]
        labels_to_show = list(range(top_n))
    else:
        labels_to_show = list(range(num_labels))
    
    if method == 'tsne':
        try:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            embeddings_2d = reducer.fit_transform(embeddings)
        except ImportError:
            print("Warning: scikit-learn not available, using PCA instead")
            method = 'pca'
    
    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        print(f"PCA explained variance: {reducer.explained_variance_ratio_.sum():.2%}")
    
    # Plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels_to_show, cmap='tab20', s=100, alpha=0.6)
    
    # Annotate some points
    for i, label in enumerate(labels_to_show[:min(20, len(labels_to_show))]):
        plt.annotate(f'T{label}', (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=8, alpha=0.7)
    
    plt.colorbar(scatter, label='Template ID')
    plt.title(f'Log Template Embeddings Visualization ({method.upper()})', 
              fontsize=14, fontweight='bold')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[+] Embedding visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_gradient_flow(model: nn.Module, save_path: Optional[str] = None):
    """
    Analyze gradient flow through the model (useful for debugging training).
    
    Args:
        model: PyTorch model
        save_path: Path to save visualization
    """
    gradients = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            gradients[name] = grad_norm
    
    if not gradients:
        print("No gradients found. Run backward() first.")
        return
    
    # Plot
    names = list(gradients.keys())
    values = list(gradients.values())
    
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(names)), values)
    plt.yticks(range(len(names)), names)
    plt.xlabel('Gradient Norm (L2)')
    plt.title('Gradient Flow Analysis', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[+] Gradient flow analysis saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_activation_distribution(model: nn.Module, sample_input: torch.Tensor, 
                                 save_path: Optional[str] = None):
    """
    Visualize activation distributions across layers.
    
    Args:
        model: PyTorch model
        sample_input: Sample input tensor
        save_path: Path to save visualization
    """
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach().cpu().numpy().flatten()
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.LSTM, nn.Linear, nn.Embedding)):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    model.eval()
    with torch.no_grad():
        _ = model(sample_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Plot
    n_layers = len(activations)
    fig, axes = plt.subplots(1, n_layers, figsize=(5*n_layers, 4))
    if n_layers == 1:
        axes = [axes]
    
    for idx, (name, acts) in enumerate(activations.items()):
        axes[idx].hist(acts, bins=50, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'{name}\nMean: {acts.mean():.3f}, Std: {acts.std():.3f}')
        axes[idx].set_xlabel('Activation Value')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Activation Distribution Across Layers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[+] Activation distribution saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_model_complexity(models_config: list, save_path: Optional[str] = None):
    """
    Compare complexity of different model configurations.
    
    Args:
        models_config: List of dicts with 'name', 'num_labels', 'embedding_dim', 
                      'hidden_size', 'num_layers'
        save_path: Path to save visualization
    """
    complexities = []
    
    for config in models_config:
        model = DeepLogLSTM(
            num_labels=config['num_labels'],
            embedding_dim=config['embedding_dim'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers']
        )
        params = count_parameters(model)
        complexities.append({
            'name': config['name'],
            'params': params['total'],
            'embedding_dim': config['embedding_dim'],
            'hidden_size': config['hidden_size'],
            'num_layers': config['num_layers']
        })
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    names = [c['name'] for c in complexities]
    params = [c['params'] for c in complexities]
    
    axes[0].barh(names, params)
    axes[0].set_xlabel('Number of Parameters')
    axes[0].set_title('Model Complexity Comparison', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Add parameter count labels
    for i, (name, p) in enumerate(zip(names, params)):
        axes[0].text(p, i, f' {p:,}', va='center')
    
    # Configuration comparison
    config_data = []
    for c in complexities:
        config_data.append([
            c['embedding_dim'],
            c['hidden_size'],
            c['num_layers']
        ])
    
    axes[1].table(cellText=config_data,
                  rowLabels=names,
                  colLabels=['Embed Dim', 'Hidden Size', 'Layers'],
                  cellLoc='center',
                  loc='center')
    axes[1].axis('off')
    axes[1].set_title('Model Configurations', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[+] Model complexity comparison saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze DeepLog LSTM Model")
    parser.add_argument("--ckpt", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--num_labels", type=int, required=True, help="Number of log templates")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory")
    
    args = parser.parse_args()
    
    # Load model
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state_dict = ckpt["model_state_dict"]
    
    embedding_dim = state_dict["embedding.weight"].shape[1]
    hidden_size = state_dict["lstm.weight_hh_l0"].shape[1]
    num_layers = 1
    
    model = DeepLogLSTM(
        num_labels=args.num_labels,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    model.load_state_dict(state_dict)
    
    # Print summary
    print_model_summary(model)
    
    # Visualize embeddings
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    visualize_embeddings(
        model, 
        args.num_labels, 
        save_path=str(output_dir / "embeddings_visualization.png")
    )
    
    print("\n[+] Model analysis completed!")
