import pickle
import argparse
import random
from pathlib import Path


def has_burst(seq, burst_len):
    run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            run += 1
            if run >= burst_len:
                return True
        else:
            run = 1
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground-truth labels using controlled anomaly injection"
    )
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--burst_len", type=int, default=6)
    parser.add_argument("--anomaly_ratio", type=float, default=0.05,
                        help="target anomaly ratio (e.g. 0.05 = 5%)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # 1. Load sequences
    with open(args.sequences, "rb") as f:
        sequences = pickle.load(f)

    print(f"[INFO] Loaded {len(sequences)} sequences")

    # 2. Find candidate anomalous sequences (burst exists)
    candidates = [
        i for i, seq in enumerate(sequences)
        if has_burst(seq, args.burst_len)
    ]

    print(f"[INFO] Burst candidates: {len(candidates)}")

    # 3. Controlled injection
    target_anomalies = int(len(sequences) * args.anomaly_ratio)
    target_anomalies = min(target_anomalies, len(candidates))

    anomaly_indices = set(random.sample(candidates, target_anomalies))

    # 4. Generate labels
    labels = [
        1 if i in anomaly_indices else 0
        for i in range(len(sequences))
    ]

    # 5. Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(labels, f)

    print("[OK] Labels generated")
    print(f"Total sequences : {len(labels)}")
    print(f"Anomalies       : {sum(labels)} ({sum(labels)/len(labels):.2%})")


if __name__ == "__main__":
    main()
