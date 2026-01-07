import torch
import torch.nn as nn


class DeepLogLSTM(nn.Module):
    """
    DeepLog-style LSTM
    - Input  : sequence of log template IDs
    - Output : logits for next log template
    """

    def __init__(
        self,
        num_labels: int,
        embedding_dim: int = 16,
        hidden_size: int = 128,
        num_layers: int = 1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(num_labels, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        """
        x: (batch_size, window_size)
        """
        x = self.embedding(x)      # (B, W, E)
        out, _ = self.lstm(x)      # (B, W, H)
        out = out[:, -1, :]        # last timestep
        logits = self.fc(out)      # (B, num_labels)
        return logits
