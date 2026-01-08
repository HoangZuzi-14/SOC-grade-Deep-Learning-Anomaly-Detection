"""
Visualization tools for model explanations
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import seaborn as sns


def plot_shap_waterfall(explanation: Dict[str, Any], save_path: Optional[str] = None):
    """
    Plot SHAP waterfall plot for single explanation
    """
    feature_importance = explanation.get("feature_importance", [])
    base_value = explanation.get("base_value", 0.0)
    score = explanation.get("score", 0.0)
    
    if not feature_importance:
        return
    
    # Prepare data
    positions = [f["position"] for f in feature_importance]
    contributions = [f["contribution"] for f in feature_importance]
    template_ids = [f["template_id"] for f in feature_importance]
    
    # Create waterfall
    cumulative = base_value
    values = [cumulative]
    labels = ["Base"]
    
    for pos, contrib, tid in zip(positions, contributions, template_ids):
        cumulative += contrib
        values.append(cumulative)
        labels.append(f"Pos {pos}\nID {tid}")
    
    values.append(score)
    labels.append("Final")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['gray'] + ['red' if c < 0 else 'blue' for c in contributions] + ['green']
    
    for i in range(len(values) - 1):
        ax.bar(i, values[i+1] - values[i], bottom=values[i], color=colors[i+1], alpha=0.7)
    
    ax.plot(range(len(values)), values, 'k--', alpha=0.5, linewidth=2)
    ax.axhline(y=base_value, color='gray', linestyle='--', alpha=0.5, label='Base Value')
    ax.axhline(y=score, color='green', linestyle='-', alpha=0.7, label='Final Score')
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('SHAP Waterfall Plot - Feature Contributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_shap_summary(explanations: List[Dict[str, Any]], save_path: Optional[str] = None):
    """
    Plot SHAP summary plot for multiple explanations
    """
    all_shap = []
    all_features = []
    
    for exp in explanations:
        if "error" not in exp:
            shap_vals = exp.get("shap_values", [])
            features = exp.get("input_sequence", [])
            if shap_vals and features:
                all_shap.extend(shap_vals[:len(features)])
                all_features.extend(features[:len(shap_vals)])
    
    if not all_shap:
        return
    
    # Create summary plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Feature importance (mean absolute SHAP)
    feature_shap = {}
    for feat, shap_val in zip(all_features, all_shap):
        if feat not in feature_shap:
            feature_shap[feat] = []
        feature_shap[feat].append(abs(shap_val))
    
    features_sorted = sorted(feature_shap.items(), key=lambda x: np.mean(x[1]), reverse=True)[:20]
    features_list = [f[0] for f in features_sorted]
    mean_shap = [np.mean(f[1]) for f in features_sorted]
    
    axes[0].barh(range(len(features_list)), mean_shap)
    axes[0].set_yticks(range(len(features_list)))
    axes[0].set_yticklabels([f"Template {f}" for f in features_list])
    axes[0].set_xlabel('Mean |SHAP Value|')
    axes[0].set_title('Feature Importance (SHAP)')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Plot 2: SHAP value distribution
    axes[1].hist(all_shap, bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    axes[1].set_xlabel('SHAP Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('SHAP Value Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_attention_heatmap(explanation: Dict[str, Any], save_path: Optional[str] = None):
    """
    Plot attention heatmap for sequence
    """
    attention_weights = explanation.get("attention_weights", [])
    input_sequence = explanation.get("input_sequence", [])
    
    if not attention_weights or not input_sequence:
        return
    
    # Create heatmap data
    data = np.array(attention_weights).reshape(1, -1)
    
    fig, ax = plt.subplots(figsize=(max(8, len(input_sequence)), 2))
    
    sns.heatmap(
        data,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        xticklabels=[f"Pos {i}\nID {tid}" for i, tid in enumerate(input_sequence)],
        yticklabels=['Attention'],
        cbar_kws={'label': 'Attention Weight'},
        ax=ax
    )
    
    ax.set_title('Attention Weights for Sequence')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_explanation_report(
    explanation: Dict[str, Any],
    save_path: Optional[str] = None
) -> str:
    """
    Create text report from explanation
    """
    report = []
    report.append("=" * 60)
    report.append("MODEL EXPLANATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    report.append(f"Sequence: {explanation.get('sequence', [])}")
    report.append(f"Target Template ID: {explanation.get('target', 'N/A')}")
    report.append(f"Anomaly Score: {explanation.get('score', 0):.4f}")
    report.append("")
    
    report.append("Feature Importance (Top Contributors):")
    report.append("-" * 60)
    feature_importance = explanation.get("feature_importance", [])[:10]
    
    for i, feat in enumerate(feature_importance, 1):
        pos = feat.get("position", "N/A")
        tid = feat.get("template_id", "N/A")
        contrib = feat.get("contribution", 0)
        report.append(f"{i:2d}. Position {pos} (Template {tid}): {contrib:+.4f}")
    
    report.append("")
    report.append("Top Predictions:")
    report.append("-" * 60)
    top_preds = explanation.get("top_predictions", [])[:5]
    for i, pred in enumerate(top_preds, 1):
        tid = pred.get("template_id", "N/A")
        prob = pred.get("probability", 0)
        report.append(f"{i}. Template {tid}: {prob*100:.2f}%")
    
    report.append("")
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    
    if save_path:
        Path(save_path).write_text(report_text, encoding='utf-8')
    
    return report_text
