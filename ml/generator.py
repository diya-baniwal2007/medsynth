import torch
from torch import nn


class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim: int = 100, num_classes: int = 2, label_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_embedding = nn.Embedding(num_classes, label_dim)
        self.project = nn.Sequential(
            nn.Linear(latent_dim + label_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        conditioned = torch.cat([noise, self.label_embedding(labels)], dim=1)
        return self.decoder(self.project(conditioned))
