import argparse
import pickle
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Case study: LSTM vs IF")
    parser.add_argument("--lstm_scores", required=True)
    parser.add_argument("--if_scores", required=True)
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    with open(args.lstm_scores, "rb") as f:
        lstm_scores = np.array(pickle.load(f), dtype=float)

    with open(args.if_scores, "rb") as f:
        if_scores = np.array(pickle.load(f), dtype=float)

    with open(args.sequences, "rb") as f:
        sequences = pickle.load(f)

    assert len(lstm_scores) == len(if_scores) == len(sequences)

    # LSTM high, IF low
    lstm_rank = np.argsort(-lstm_scores)
    if_rank = np.argsort(-if_scores)

    if_rank_pos = {idx: rank for rank, idx in enumerate(if_rank)}

    selected = []
    for idx in lstm_rank:
        if if_rank_pos[idx] > len(if_scores) * 0.7:
            selected.append(idx)
        if len(selected) >= args.topk:
            break

    print("=== Case Study: LSTM-high / IF-low Sequences ===")

    for i, idx in enumerate(selected, 1):
        print(f"\n--- Case {i} ---")
        print(f"Index       : {idx}")
        print(f"LSTM score  : {lstm_scores[idx]:.4f}")
        print(f"IF score    : {if_scores[idx]:.4f}")
        print(f"Sequence    : {sequences[idx]}")


if __name__ == "__main__":
    main()
