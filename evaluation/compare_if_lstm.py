import argparse
import pickle
import numpy as np
from scipy.stats import spearmanr


def main():
    parser = argparse.ArgumentParser(description="Compare IF vs LSTM scores (unsupervised)")
    parser.add_argument("--lstm_scores", required=True, help="lstm_scores.pkl (NLL)")
    parser.add_argument("--if_scores", required=True, help="if_scores.pkl")
    parser.add_argument("--topk_ratio", type=float, default=0.05, help="top-k ratio (e.g. 0.05 = 5%)")
    args = parser.parse_args()

    with open(args.lstm_scores, "rb") as f:
        lstm_scores = np.array(pickle.load(f), dtype=float)

    with open(args.if_scores, "rb") as f:
        if_scores = np.array(pickle.load(f), dtype=float)

    assert len(lstm_scores) == len(if_scores), "Score length mismatch"

    n = len(lstm_scores)
    k = int(n * args.topk_ratio)

    # ranking (higher = more anomalous)
    lstm_rank = np.argsort(-lstm_scores)
    if_rank = np.argsort(-if_scores)

    lstm_topk = set(lstm_rank[:k])
    if_topk = set(if_rank[:k])

    intersection = lstm_topk & if_topk
    union = lstm_topk | if_topk

    jaccard = len(intersection) / len(union)
    spearman_corr, _ = spearmanr(lstm_scores, if_scores)

    print("=== Unsupervised IF vs LSTM Comparison ===")
    print(f"Samples        : {n}")
    print(f"Top-k ratio    : {args.topk_ratio:.2%}")
    print(f"Top-k size     : {k}")
    print(f"Overlap        : {len(intersection)}")
    print(f"Jaccard index  : {jaccard:.4f}")
    print(f"Spearman corr  : {spearman_corr:.4f}")

    if jaccard < 0.5:
        print("\n[INSIGHT] LSTM and IF capture different anomaly signals.")
    else:
        print("\n[INSIGHT] LSTM and IF largely agree on anomalous samples.")


if __name__ == "__main__":
    main()
