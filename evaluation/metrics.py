import argparse
import pickle
import numpy as np

print("[INFO] metrics.py loaded")

try:
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
    )
except Exception as e:
    print("[FATAL] sklearn import failed:", e)
    exit(1)


def main():
    print("[INFO] metrics.py started")

    parser = argparse.ArgumentParser(description="Anomaly detection metrics")
    parser.add_argument("--scored", required=True)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()

    print("[INFO] loading scored file")
    with open(args.scored, "rb") as f:
        scored = pickle.load(f)

    scores = np.array(scored["scores_norm"])
    preds = np.array(scored["preds"])

    print("[INFO] loading labels")
    with open(args.labels, "rb") as f:
        y_true = np.array(pickle.load(f))

    print(f"[INFO] samples: scores={len(scores)}, labels={len(y_true)}")
    print(f"[INFO] label distribution: {dict(zip(*np.unique(y_true, return_counts=True)))}")
    print(f"[INFO] pred distribution : {dict(zip(*np.unique(preds, return_counts=True)))}")

    if len(np.unique(y_true)) < 2:
        print("[ERROR] Only one class in labels → cannot compute metrics")
        return

    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    roc_auc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)

    print("\n=== Evaluation Metrics ===")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")


if __name__ == "__main__":
    main()
