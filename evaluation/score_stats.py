import argparse
import pickle
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="LSTM intrinsic score statistics")
    parser.add_argument("--scores", required=True, help="lstm_scores.pkl (NLL)")
    args = parser.parse_args()

    with open(args.scores, "rb") as f:
        scores = np.array(pickle.load(f), dtype=float)

    print("=== LSTM Score Statistics (NLL) ===")
    print(f"Total samples : {len(scores)}")
    print(f"Mean          : {scores.mean():.4f}")
    print(f"Std           : {scores.std():.4f}")
    print(f"Min           : {scores.min():.4f}")
    print(f"Max           : {scores.max():.4f}")

    for p in [90, 95, 99]:
        print(f"P{p:<2}          : {np.percentile(scores, p):.4f}")

    # tail analysis
    tail_1 = scores >= np.percentile(scores, 99)
    tail_5 = scores >= np.percentile(scores, 95)

    print("\n=== Tail Analysis ===")
    print(f"Top 1% samples : {tail_1.sum()}")
    print(f"Top 5% samples : {tail_5.sum()}")


if __name__ == "__main__":
    main()
