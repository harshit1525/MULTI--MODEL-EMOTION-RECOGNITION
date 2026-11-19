import torch
import torch.nn as nn

class VideoClassifier(nn.Module):
    """
    MLP classifier for 2048-D ResNet-50 visual feature embeddings.
    """
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 256, num_classes: int = 7, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

