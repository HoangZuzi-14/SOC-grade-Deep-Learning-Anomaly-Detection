import pickle
import argparse
import numpy as np
from sklearn.ensemble import IsolationForest


def main():
    parser = argparse.ArgumentParser(description="Run Isolation Forest baseline")
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_features", type=float, default=1.0)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    # Load sequences
    with open(args.sequences, "rb") as f:
        sequences = pickle.load(f)

    # Feature: bag-of-templates
    max_tid = max(max(seq) for seq in sequences) + 1
    X = np.zeros((len(sequences), max_tid), dtype=np.float32)

    for i, seq in enumerate(sequences):
        for t in seq:
            X[i, t] += 1

    # Isolation Forest
    clf = IsolationForest(
        n_estimators=args.n_estimators,
        max_features=args.max_features,
        contamination="auto",
        random_state=args.random_state,
        n_jobs=-1,
    )
    clf.fit(X)

    # decision_function: higher = more normal
    scores = -clf.decision_function(X)

    # Save scores
    with open(args.out, "wb") as f:
        pickle.dump(scores, f)

    print("[OK] Isolation Forest scores generated")
    print(f"Samples : {len(scores)}")
    print(f"Min     : {scores.min():.4f}")
    print(f"Max     : {scores.max():.4f}")


if __name__ == "__main__":
    main()
