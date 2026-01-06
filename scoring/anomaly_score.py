import argparse
import pickle
import numpy as np
from pathlib import Path


def normalize_minmax(scores: np.ndarray) -> np.ndarray:
    min_s = scores.min()
    max_s = scores.max()
    if max_s == min_s:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)


def compute_threshold(scores: np.ndarray, percentile: float) -> float:
    return float(np.percentile(scores, percentile))


def main():
    parser = argparse.ArgumentParser(description="Anomaly scoring & detection")
    parser.add_argument("--scores", required=True, help="raw score pkl file")
    parser.add_argument("--out", required=True, help="output scored pkl")
    parser.add_argument("--percentile", type=float, default=95.0)
    args = parser.parse_args()

    # 1. load raw scores
    with open(args.scores, "rb") as f:
        raw_scores = np.array(pickle.load(f), dtype=float)

    # 2. normalize
    norm_scores = normalize_minmax(raw_scores)

    # 3. threshold
    threshold = compute_threshold(norm_scores, args.percentile)

    # 4. detect anomaly
    preds = (norm_scores >= threshold).astype(int)

    result = {
        "scores_raw": raw_scores,
        "scores_norm": norm_scores,
        "threshold": float(threshold),
        "percentile": args.percentile,
        "preds": preds,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(result, f)

    print("=== Scoring Result ===")
    print(f"Total samples : {len(preds)}")
    print(f"Threshold     : P{args.percentile} = {threshold:.4f}")
    print(f"Anomalies     : {preds.sum()}")


if __name__ == "__main__":
    main()
