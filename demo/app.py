# demo/app.py
from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scoring.anomaly_score import compute_thresholds, score_to_soc_decision, Thresholds


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_event_mapping(path: str) -> Dict[int, str]:
    """
    Try to load template_id -> template_text mapping from event_mapping.json.
    Handles a few common shapes robustly.
    """
    p = Path(path)
    if not p.exists():
        return {}

    obj = json.loads(p.read_text(encoding="utf-8"))

    # Case 1: {"0": "...", "1": "..."} or {0:"..."}
    if isinstance(obj, dict):
        # maybe nested like {"id_to_template": {...}}
        for k in ["id_to_template", "templates", "mapping", "event_mapping"]:
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break

        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                try:
                    kid = int(k)
                except Exception:
                    continue
                if isinstance(v, str):
                    out[kid] = v
                elif isinstance(v, dict):
                    # maybe {"template": "..."} or {"template_text":"..."}
                    out[kid] = str(v.get("template_text") or v.get("template") or v)
                else:
                    out[kid] = str(v)
            return out

    # Case 2: list of templates where index is id
    if isinstance(obj, list):
        out = {}
        for i, v in enumerate(obj):
            out[int(i)] = v if isinstance(v, str) else str(v)
        return out

    return {}


def format_seq(seq: List[int], mapping: Dict[int, str], max_events: int = 8) -> str:
    tail = seq[-max_events:]
    if not mapping:
        return " ".join(str(x) for x in tail)
    # show id:short_text
    parts = []
    for tid in tail:
        txt = mapping.get(int(tid), "")
        txt = txt.replace("\n", " ").strip()
        if len(txt) > 40:
            txt = txt[:37] + "..."
        parts.append(f"{tid}:{txt}" if txt else str(tid))
    return " | ".join(parts)


def pick_indices(scores: List[float], top_n: int, random_n: int, seed: int) -> List[int]:
    n = len(scores)
    idx_sorted = sorted(range(n), key=lambda i: scores[i], reverse=True)
    top = idx_sorted[: min(top_n, n)]
    rnd_pool = [i for i in range(n) if i not in set(top)]
    random.seed(seed)
    rnd = random.sample(rnd_pool, k=min(random_n, len(rnd_pool))) if rnd_pool and random_n > 0 else []
    return top + rnd


def print_block(
    model_name: str,
    indices: List[int],
    sequences: List[List[int]],
    scores: List[float],
    th: Thresholds,
    mapping: Dict[int, str],
    min_alert: str,
    max_events: int,
):
    print("=" * 100)
    print(f"MODEL: {model_name}")
    print(f"THRESHOLDS: p95={th.p95:.6f}  p99={th.p99:.6f}  p999={th.p999:.6f}   (alert if >= {min_alert})")
    print("-" * 100)
    header = f"{'idx':>6}  {'score':>10}  {'alert':>5}  {'sev':>4}  sequence_tail"
    print(header)
    print("-" * 100)
    for i in indices:
        score = float(scores[i])
        alert, sev = score_to_soc_decision(score, th, min_alert=min_alert)
        seq_tail = format_seq(sequences[i], mapping, max_events=max_events)
        print(f"{i:>6}  {score:>10.6f}  {str(alert):>5}  {sev:>4}  {seq_tail}")
    print()


def main():
    ap = argparse.ArgumentParser(description="SOC demo CLI: show anomaly alerts from LSTM and IsolationForest scores.")
    ap.add_argument("--sequences", default="data/sequences/sequences.pkl")
    ap.add_argument("--lstm_scores", default="data/sequences/lstm_scores.pkl")
    ap.add_argument("--if_scores", default="data/sequences/if_scores.pkl")
    ap.add_argument("--event_mapping", default="data/sequences/event_mapping.json")
    ap.add_argument("--model", choices=["lstm", "if", "both"], default="both")
    ap.add_argument("--top_n", type=int, default=10, help="Show top-N highest scores.")
    ap.add_argument("--random_n", type=int, default=5, help="Show random samples (likely normal).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_alert", choices=["LOW", "MED", "HIGH"], default="MED")
    ap.add_argument("--max_events", type=int, default=8, help="How many last events to print per sequence.")
    args = ap.parse_args()

    sequences: List[List[int]] = load_pickle(args.sequences)
    mapping = load_event_mapping(args.event_mapping)

    if args.model in ("lstm", "both"):
        lstm_scores: List[float] = load_pickle(args.lstm_scores)
        if len(lstm_scores) != len(sequences):
            raise ValueError(f"Length mismatch: lstm_scores={len(lstm_scores)} vs sequences={len(sequences)}")

        th_lstm = compute_thresholds(lstm_scores)
        idx_lstm = pick_indices(lstm_scores, top_n=args.top_n, random_n=args.random_n, seed=args.seed)
        print_block("LSTM (NLL)", idx_lstm, sequences, lstm_scores, th_lstm, mapping, args.min_alert, args.max_events)

    if args.model in ("if", "both"):
        if_scores: List[float] = load_pickle(args.if_scores)
        if len(if_scores) != len(sequences):
            raise ValueError(f"Length mismatch: if_scores={len(if_scores)} vs sequences={len(sequences)}")

        th_if = compute_thresholds(if_scores)
        idx_if = pick_indices(if_scores, top_n=args.top_n, random_n=args.random_n, seed=args.seed)
        print_block("IsolationForest", idx_if, sequences, if_scores, th_if, mapping, args.min_alert, args.max_events)


if __name__ == "__main__":
    main()
