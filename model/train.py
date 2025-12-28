import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from deeplog_lstm import DeepLogLSTM

class SlidingWindowDataset(Dataset):
    """
    sequences.pkl format:
    [
        [t1, t2, t3, ..., tN],
        ...
    ]
    """

    def __init__(self, sequences, window_size):
        self.samples = []

        for seq in sequences:
            if len(seq) <= window_size:
                continue

            for i in range(len(seq) - window_size):
                X = seq[i : i + window_size]
                y = seq[i + window_size]
                self.samples.append((X, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        return (
            torch.tensor(X, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )



# Training Loop
def train(model, dataloader, optimizer, criterion, device, epochs):
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"Loss: {total_loss / len(dataloader):.4f}"
        )


# Main
if __name__ == "__main__":

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    WINDOW_SIZE = 5     # m chỉnh cho phù hợp


    path = os.path.join("..", "data", "sequences","sequences.pkl")

    with open(path, "rb") as f:
        sequences = pickle.load(f)

    print(f"[+] Loaded {len(sequences)} sequences")

    # Infer number of labels
    all_labels = set()
    for seq in sequences:
        all_labels.update(seq)

    num_labels = max(all_labels) + 1

    print(f"[+] Num labels : {num_labels}")
    print(f"[+] Window size: {WINDOW_SIZE}")

    # Dataset & Dataloader
    dataset = SlidingWindowDataset(sequences, WINDOW_SIZE)
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        drop_last=True
    )

    print(f"[+] Training samples: {len(dataset)}")

    # Model
    model = DeepLogLSTM(
        num_labels=num_labels,
        embedding_dim=16,
        hidden_size=128,
        num_layers=1
    ).to(DEVICE)

    # Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        epochs=10
    )

    # Save model weight
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_labels": num_labels,
            "window_size": WINDOW_SIZE,
        },
        "model.pth"
    )

    print("[+] Model saved: model.pth")
