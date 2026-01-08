import os
import pickle
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Import DeepLogLSTM - handle both relative and absolute imports
try:
    from .deeplog_lstm import DeepLogLSTM
except ImportError:
    try:
        from model.deeplog_lstm import DeepLogLSTM
    except ImportError:
        from deeplog_lstm import DeepLogLSTM

class SlidingWindowDataset(Dataset):
    """
    sequences.pkl format:
    [
        [t1, t2, t3, ..., tN],
        ...
    ]
    """

    def __init__(self, sequences, window_size):
        self.samples = []

        for seq in sequences:
            if len(seq) <= window_size:
                continue

            for i in range(len(seq) - window_size):
                X = seq[i : i + window_size]
                y = seq[i + window_size]
                self.samples.append((X, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        return (
            torch.tensor(X, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )



# Training Loop with Metrics Tracking
def train(model, train_loader, val_loader, test_loader, optimizer, criterion, device, epochs, save_dir):
    """
    Training loop with validation, metrics tracking, and visualization.
    """
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_topk_accs = []
    val_topk_accs = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_topk_correct = 0
        train_total = 0
        
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
            
            # Top-k accuracy (k=5)
            _, topk_pred = torch.topk(logits, k=min(5, logits.size(1)), dim=1)
            train_topk_correct += topk_pred.eq(y.view(-1, 1).expand_as(topk_pred)).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_topk_acc = 100 * train_topk_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_topk_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                
                logits = model(X)
                loss = criterion(logits, y)
                
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(logits.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
                
                # Top-k accuracy
                _, topk_pred = torch.topk(logits, k=min(5, logits.size(1)), dim=1)
                val_topk_correct += topk_pred.eq(y.view(-1, 1).expand_as(topk_pred)).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_topk_acc = 100 * val_topk_correct / val_total
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_topk_accs.append(train_topk_acc)
        val_topk_accs.append(val_topk_acc)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val_loss,
                "val_acc": val_acc,
            }, os.path.join(save_dir, "best_model.pth"))
        
        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
            f"Train Top-5: {train_topk_acc:.2f}% | Val Top-5: {val_topk_acc:.2f}%"
        )
    
    # Save training history
    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs,
        "train_topk_acc": train_topk_accs,
        "val_topk_acc": val_topk_accs,
    }
    
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plot_training_curves(history, save_dir)
    
    # Evaluate on test set
    print("\n[*] Evaluating on test set...")
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_topk_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            
            logits = model(X)
            loss = criterion(logits, y)
            
            test_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            test_total += y.size(0)
            test_correct += (predicted == y).sum().item()
            
            # Top-k accuracy
            _, topk_pred = torch.topk(logits, k=min(5, logits.size(1)), dim=1)
            test_topk_correct += topk_pred.eq(y.view(-1, 1).expand_as(topk_pred)).sum().item()
    
    avg_test_loss = test_loss / len(test_loader)
    test_acc = 100 * test_correct / test_total
    test_topk_acc = 100 * test_topk_correct / test_total
    
    print(f"[+] Test Results:")
    print(f"    Test Loss: {avg_test_loss:.4f}")
    print(f"    Test Acc: {test_acc:.2f}%")
    print(f"    Test Top-5 Acc: {test_topk_acc:.2f}%")
    
    # Add test metrics to history
    history["test_loss"] = avg_test_loss
    history["test_acc"] = test_acc
    history["test_topk_acc"] = test_topk_acc
    
    # Save updated history
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    return history


def plot_training_curves(history, save_dir):
    """Visualize training curves for loss and accuracy."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, history["train_loss"], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history["val_loss"], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(epochs, history["train_acc"], 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, history["val_acc"], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Model Accuracy (Top-1)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top-k Accuracy curves
    axes[1, 0].plot(epochs, history["train_topk_acc"], 'b-', label='Training Top-5 Acc', linewidth=2)
    axes[1, 0].plot(epochs, history["val_topk_acc"], 'r-', label='Validation Top-5 Acc', linewidth=2)
    axes[1, 0].set_title('Model Top-5 Accuracy', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Top-5 Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined view
    ax2 = axes[1, 0].twinx()
    ax2.plot(epochs, history["train_loss"], 'g--', alpha=0.5, label='Train Loss')
    ax2.plot(epochs, history["val_loss"], 'orange', linestyle='--', alpha=0.5, label='Val Loss')
    ax2.set_ylabel('Loss', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    
    # Learning curve (loss difference)
    train_val_diff = [t - v for t, v in zip(history["train_loss"], history["val_loss"])]
    axes[1, 1].plot(epochs, train_val_diff, 'purple', linewidth=2)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1, 1].set_title('Train-Val Loss Difference (Overfitting Indicator)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Difference')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=300, bbox_inches='tight')
    print(f"[+] Training curves saved to {os.path.join(save_dir, 'training_curves.png')}")
    plt.close()


# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepLog LSTM for Anomaly Detection")
    parser.add_argument("--sequences", type=str, default="../data/sequences/sequences.pkl",
                        help="Path to sequences.pkl")
    parser.add_argument("--window_size", type=int, default=5,
                        help="Sliding window size")
    parser.add_argument("--embedding_dim", type=int, default=16,
                        help="Embedding dimension")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="LSTM hidden size")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="Number of LSTM layers")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="Validation split ratio")
    parser.add_argument("--test_split", type=float, default=0.15,
                        help="Test split ratio")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu), auto-detect if None")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Output directory for model and plots")
    
    args = parser.parse_args()
    
    # Device setup
    if args.device is None:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = args.device
    
    print(f"[+] Using device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"[+] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[+] CUDA Version: {torch.version.cuda}")
    
    # Load sequences - handle both relative and absolute paths
    path = args.sequences
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if not os.path.isabs(path):
        # Try multiple path resolutions
        possible_paths = []
        
        # 1. If path starts with ../, resolve from project root
        if path.startswith('../'):
            possible_paths.append(os.path.join(project_root, path.replace('../', '')))
        # 2. Try relative to project root (for paths like "data/sequences/sequences.pkl")
        possible_paths.append(os.path.join(project_root, path))
        # 3. Try relative to model directory
        possible_paths.append(os.path.join(os.path.dirname(__file__), path))
        
        # Find first existing path
        path = None
        for p in possible_paths:
            p = os.path.normpath(p)
            if os.path.exists(p):
                path = p
                break
        
        if path is None:
            raise FileNotFoundError(
                f"Sequences file not found. Tried:\n"
                f"  - {os.path.join(project_root, args.sequences)}\n"
                f"  - {os.path.join(os.path.dirname(__file__), args.sequences)}\n"
                f"Expected location: {os.path.join(project_root, 'data', 'sequences', 'sequences.pkl')}"
            )
    else:
        # Absolute path
        path = os.path.normpath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Sequences file not found: {path}")
    
    with open(path, "rb") as f:
        sequences = pickle.load(f)
    
    print(f"[+] Loaded {len(sequences)} sequences")
    
    # Infer number of labels
    all_labels = set()
    for seq in sequences:
        all_labels.update(seq)
    
    num_labels = max(all_labels) + 1
    
    print(f"[+] Model Configuration:")
    print(f"    Num labels    : {num_labels}")
    print(f"    Window size   : {args.window_size}")
    print(f"    Embedding dim : {args.embedding_dim}")
    print(f"    Hidden size   : {args.hidden_size}")
    print(f"    Num layers    : {args.num_layers}")
    print(f"    Batch size    : {args.batch_size}")
    print(f"    Learning rate : {args.lr}")
    print(f"    Epochs        : {args.epochs}")
    
    # Create dataset
    dataset = SlidingWindowDataset(sequences, args.window_size)
    print(f"[+] Total samples: {len(dataset)}")
    
    # Train/Val/Test split
    test_size = int(len(dataset) * args.test_split)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size - test_size
    
    # Ensure we have enough data
    if train_size < 1:
        raise ValueError(f"Not enough data for train/val/test split. Total: {len(dataset)}, "
                        f"Requested: train={train_size}, val={val_size}, test={test_size}")
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"[+] Train samples: {len(train_dataset)} ({len(train_dataset)/len(dataset)*100:.1f}%)")
    print(f"[+] Val samples  : {len(val_dataset)} ({len(val_dataset)/len(dataset)*100:.1f}%)")
    print(f"[+] Test samples : {len(test_dataset)} ({len(test_dataset)/len(dataset)*100:.1f}%)")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    
    # Model
    model = DeepLogLSTM(
        num_labels=num_labels,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    ).to(DEVICE)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] Model Parameters:")
    print(f"    Total      : {total_params:,}")
    print(f"    Trainable  : {trainable_params:,}")
    
    # Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Train
    print(f"\n[+] Starting training...")
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        epochs=args.epochs,
        save_dir=output_dir
    )
    
    # Save final model
    model_path = os.path.join(output_dir, "model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_labels": num_labels,
            "window_size": args.window_size,
            "embedding_dim": args.embedding_dim,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "final_train_acc": history["train_acc"][-1],
            "final_val_acc": history["val_acc"][-1],
        },
        model_path
    )
    
    print(f"\n[+] Training completed!")
    print(f"[+] Model saved: {model_path}")
    print(f"[+] Best model saved: {os.path.join(output_dir, 'best_model.pth')}")
    print(f"[+] Training history: {os.path.join(output_dir, 'training_history.json')}")
    print(f"[+] Final metrics:")
    print(f"    Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f} | Test Loss: {history.get('test_loss', 0):.4f}")
    print(f"    Train Acc: {history['train_acc'][-1]:.2f}% | Val Acc: {history['val_acc'][-1]:.2f}% | Test Acc: {history.get('test_acc', 0):.2f}%")
