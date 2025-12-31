# model/infer.py
import argparse
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F

from deeplog_lstm import DeepLogLSTM


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", default="data/sequences/sequences.pkl")
    ap.add_argument("--ckpt", default="model/model.pth")
    ap.add_argument("--out", default="data/sequences/lstm_scores.pkl")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--score_type", choices=["nll", "topk"], default="nll")
    ap.add_argument("--topk", type=int, default=9)
    args = ap.parse_args()

    device = args.device

    # 1) load sequences
    with open(args.sequences, "rb") as f:
        sequences = pickle.load(f)

    # 2) load checkpoint A
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt["model_state_dict"]
    num_labels = int(ckpt["num_labels"])
    window_size = int(ckpt["window_size"])

    # (optional) derive dims from checkpoint to avoid mismatch
    embedding_dim = state_dict["embedding.weight"].shape[1]      # e.g., 16
    hidden_size = state_dict["lstm.weight_hh_l0"].shape[1]       # e.g., 128
    num_layers = 1  # based on presence of only l0 in your checkpoint

    # 3) build model + load weights
    model = DeepLogLSTM(
        num_labels=num_labels,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 4) infer score per sequence
    scores = []
    for seq in sequences:
        if not seq or len(seq) < 2:
            scores.append(0.0)
            continue

        # enforce window_size like training expects
        if len(seq) > window_size:
            seq = seq[-window_size:]
        elif len(seq) < window_size:
            # pad left with 0 (chỉ OK nếu 0 là template thật ít ảnh hưởng; nếu project bạn có pad_id riêng thì dùng pad_id)
            pad_len = window_size - len(seq)
            seq = ([0] * pad_len) + seq

        x = torch.tensor(seq[:-1], dtype=torch.long, device=device).unsqueeze(0)  # (1, W-1)
        y = torch.tensor(seq[-1], dtype=torch.long, device=device)                # ()

        logits = model(x).squeeze(0)  # (C,)

        if args.score_type == "topk":
            k = min(args.topk, logits.numel())
            topk_ids = torch.topk(logits, k=k).indices
            score = 0.0 if (y in topk_ids) else 1.0
        else:
            logp = F.log_softmax(logits, dim=-1)
            score = float((-logp[y]).item())  # NLL

        scores.append(score)

    # 5) save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(scores, f)

    print(f"[OK] sequences={len(sequences)} scores={len(scores)} saved={args.out}")
    print(f"num_labels={num_labels} window_size={window_size} embed_dim={embedding_dim} hidden_size={hidden_size}")


if __name__ == "__main__":
    main()
