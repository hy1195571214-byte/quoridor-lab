"""Policy architectures shared by training and saved-checkpoint inference."""
from torch import nn


def network(architecture="mlp"):
    if architecture == "mlp":
        return nn.Sequential(nn.Flatten(), nn.Linear(648, 256), nn.ReLU(), nn.Linear(256, 209))
    if architecture == "cnn32":
        return nn.Sequential(
            nn.Conv2d(8, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 9 * 9, 209),
        )
    raise ValueError(f"Unknown architecture: {architecture}")
