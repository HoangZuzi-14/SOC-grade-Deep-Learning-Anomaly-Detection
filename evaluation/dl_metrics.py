"""
Deep Learning Specific Evaluation Metrics
Provides metrics tailored for sequence-based anomaly detection models.
"""

import pickle
import numpy as np
import argparse
from typing import Dict, Optional
import matplotlib.pyplot as plt
from pathlib import Path


def calculate_topk_accuracy(predictions: np.ndarray, targets: np.ndarray, k: int = 5) -> float:
    """
    Calculate top-k accuracy.
    
    Args:
        predictions: (N, num_classes) array of prediction logits/probabilities
        targets: (N,) array of true labels
        k: Top-k value
    
    Returns:
        Top-k accuracy as float
    """
    topk_preds = np.argsort(predictions, axis=1)[:, -k:]
    correct = np.sum([target in topk_preds[i] for i, target in enumerate(targets)])
    return correct / len(targets)


def calculate_perplexity(scores: np.ndarray) -> float:
    """
    Calculate perplexity from negative log-likelihood scores.
    
    Args:
        scores: Array of NLL scores
    
    Returns:
        Perplexity value
    """
    # Perplexity = exp(mean(NLL))
    return np.exp(np.mean(scores))


def calculate_anomaly_detection_metrics(scores: np.ndarray, threshold_percentile: float = 95) -> Dict:
    """
    Calculate anomaly detection metrics using percentile threshold.
    
    Args:
        scores: Anomaly scores (higher = more anomalous)
        threshold_percentile: Percentile to use as threshold
    
    Returns:
        Dictionary with metrics
    """
    threshold = np.percentile(scores, threshold_percentile)
    predictions = (scores >= threshold).astype(int)
    
    # In unsupervised setting, we can't calculate precision/recall without ground truth
    # But we can provide statistics
    n_anomalies = predictions.sum()
    n_normal = len(predictions) - n_anomalies
    anomaly_ratio = n_anomalies / len(predictions)
    
    return {
        'threshold': threshold,
        'n_anomalies': n_anomalies,
        'n_normal': n_normal,
        'anomaly_ratio': anomaly_ratio,
        'threshold_percentile': threshold_percentile
    }


def compare_model_performance(lstm_scores: np.ndarray, if_scores: np.ndarray, 
                              save_path: Optional[str] = None) -> Dict:
    """
    Comprehensive comparison between LSTM and Isolation Forest.
    
    Args:
        lstm_scores: LSTM anomaly scores
        if_scores: Isolation Forest anomaly scores
        save_path: Path to save comparison plots
    
    Returns:
        Dictionary with comparison metrics
    """
    from scipy.stats import spearmanr, pearsonr
    
    # Normalize scores to [0, 1] for comparison
    def normalize(s):
        s_min, s_max = s.min(), s.max()
        return (s - s_min) / (s_max - s_min + 1e-8)
    
    lstm_norm = normalize(lstm_scores)
    if_norm = normalize(if_scores)
    
    # Correlation
    spearman_corr, spearman_p = spearmanr(lstm_scores, if_scores)
    pearson_corr, pearson_p = pearsonr(lstm_scores, if_scores)
    
    # Top-k overlap
    k = int(len(lstm_scores) * 0.05)  # Top 5%
    lstm_topk = set(np.argsort(-lstm_scores)[:k])
    if_topk = set(np.argsort(-if_scores)[:k])
    
    intersection = lstm_topk & if_topk
    union = lstm_topk | if_topk
    jaccard = len(intersection) / len(union) if len(union) > 0 else 0
    
    metrics = {
        'spearman_correlation': spearman_corr,
        'spearman_pvalue': spearman_p,
        'pearson_correlation': pearson_corr,
        'pearson_pvalue': pearson_p,
        'jaccard_index': jaccard,
        'topk_overlap': len(intersection),
        'topk_size': k
    }
    
    # Visualization
    if save_path:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Score distribution comparison
        axes[0, 0].hist(lstm_scores, bins=50, alpha=0.6, label='LSTM', color='blue')
        axes[0, 0].hist(if_scores, bins=50, alpha=0.6, label='IF', color='red')
        axes[0, 0].set_xlabel('Anomaly Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Score Distribution Comparison', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[0, 1].scatter(lstm_norm, if_norm, alpha=0.3, s=10)
        axes[0, 1].set_xlabel('LSTM Score (normalized)')
        axes[0, 1].set_ylabel('IF Score (normalized)')
        axes[0, 1].set_title(f'Score Correlation\nSpearman: {spearman_corr:.3f}, Pearson: {pearson_corr:.3f}', 
                            fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top-k comparison
        topk_data = {
            'LSTM only': len(lstm_topk - if_topk),
            'IF only': len(if_topk - lstm_topk),
            'Both': len(intersection)
        }
        axes[1, 0].bar(topk_data.keys(), topk_data.values(), color=['blue', 'red', 'green'])
        axes[1, 0].set_ylabel('Number of Samples')
        axes[1, 0].set_title(f'Top-{k} Anomaly Overlap\nJaccard Index: {jaccard:.3f}', 
                            fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Rank comparison
        lstm_ranks = np.argsort(-lstm_scores)
        if_ranks = np.argsort(-if_scores)
        rank_correlation = np.corrcoef(lstm_ranks, if_ranks)[0, 1]
        
        axes[1, 1].scatter(lstm_ranks[:k], if_ranks[:k], alpha=0.5, s=20)
        axes[1, 1].set_xlabel('LSTM Rank')
        axes[1, 1].set_ylabel('IF Rank')
        axes[1, 1].set_title(f'Top-{k} Rank Comparison\nCorrelation: {rank_correlation:.3f}', 
                            fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[+] Model comparison plot saved to {save_path}")
        plt.close()
    
    return metrics


def analyze_score_distribution(scores: np.ndarray, model_name: str = "Model", 
                               save_path: Optional[str] = None) -> Dict:
    """
    Detailed analysis of anomaly score distribution.
    
    Args:
        scores: Anomaly scores
        model_name: Name of the model
        save_path: Path to save visualization
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores),
        'median': np.median(scores),
        'q25': np.percentile(scores, 25),
        'q75': np.percentile(scores, 75),
        'q90': np.percentile(scores, 90),
        'q95': np.percentile(scores, 95),
        'q99': np.percentile(scores, 99),
        'skewness': float(np.mean(((scores - np.mean(scores)) / np.std(scores)) ** 3)),
        'kurtosis': float(np.mean(((scores - np.mean(scores)) / np.std(scores)) ** 4))
    }
    
    if save_path:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histogram
        axes[0, 0].hist(scores, bins=100, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(stats['mean'], color='red', linestyle='--', label=f"Mean: {stats['mean']:.4f}")
        axes[0, 0].axvline(stats['median'], color='green', linestyle='--', label=f"Median: {stats['median']:.4f}")
        axes[0, 0].set_xlabel('Anomaly Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'{model_name} Score Distribution', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot
        axes[0, 1].boxplot(scores, vert=True)
        axes[0, 1].set_ylabel('Anomaly Score')
        axes[0, 1].set_title(f'{model_name} Score Box Plot', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Q-Q plot (check normality)
        from scipy import stats as scipy_stats
        scipy_stats.probplot(scores, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal Distribution)', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_scores = np.sort(scores)
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        axes[1, 1].plot(sorted_scores, cumulative, linewidth=2)
        axes[1, 1].axvline(stats['q95'], color='red', linestyle='--', alpha=0.7, label='95th percentile')
        axes[1, 1].axvline(stats['q99'], color='orange', linestyle='--', alpha=0.7, label='99th percentile')
        axes[1, 1].set_xlabel('Anomaly Score')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Cumulative Distribution Function', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - Comprehensive Score Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[+] Score distribution analysis saved to {save_path}")
        plt.close()
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Deep Learning Evaluation Metrics")
    parser.add_argument("--lstm_scores", type=str, required=True, help="LSTM scores pickle file")
    parser.add_argument("--if_scores", type=str, help="Isolation Forest scores pickle file")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory")
    
    args = parser.parse_args()
    
    # Load scores
    with open(args.lstm_scores, "rb") as f:
        lstm_scores = np.array(pickle.load(f))
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("DEEP LEARNING EVALUATION METRICS")
    print("=" * 80)
    
    # LSTM analysis
    print("\n[LSTM MODEL ANALYSIS]")
    lstm_stats = analyze_score_distribution(
        lstm_scores, 
        model_name="LSTM",
        save_path=str(output_dir / "lstm_score_analysis.png")
    )
    
    for key, value in lstm_stats.items():
        print(f"  {key:15s}: {value:.4f}")
    
    # Perplexity
    perplexity = calculate_perplexity(lstm_scores)
    print(f"\n  Perplexity        : {perplexity:.4f}")
    
    # Anomaly detection metrics
    anomaly_metrics = calculate_anomaly_detection_metrics(lstm_scores, threshold_percentile=95)
    print(f"\n  Anomaly Threshold (95th percentile): {anomaly_metrics['threshold']:.4f}")
    print(f"  Detected Anomalies: {anomaly_metrics['n_anomalies']} ({anomaly_metrics['anomaly_ratio']:.2%})")
    
    # Comparison with IF if available
    if args.if_scores:
        with open(args.if_scores, "rb") as f:
            if_scores = np.array(pickle.load(f))
        
        print("\n[MODEL COMPARISON: LSTM vs Isolation Forest]")
        comparison = compare_model_performance(
            lstm_scores,
            if_scores,
            save_path=str(output_dir / "model_comparison.png")
        )
        
        for key, value in comparison.items():
            if isinstance(value, float):
                print(f"  {key:25s}: {value:.4f}")
            else:
                print(f"  {key:25s}: {value}")
    
    print("\n" + "=" * 80)
    print("[+] Evaluation completed!")


if __name__ == "__main__":
    main()
