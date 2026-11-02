import torch
from torch import nn


class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes: int = 2, image_size: int = 28):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.15),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.15),
            nn.Flatten(),
        )
        self.image_head = nn.Linear(64 * 7 * 7, 1)
        self.label_embedding = nn.Embedding(num_classes, 64 * 7 * 7)

    def forward(self, image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = self.features(image)
        projection = (features * self.label_embedding(labels)).sum(dim=1, keepdim=True)
        return self.image_head(features) + projection / features.shape[1] ** 0.5
