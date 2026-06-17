"""
FaceRig Denoiser — сеть для очистки активаций лицевых мышц от шума MediaPipe.

Архитектура:
  Энкодер: input_dim → 64 → 32 → latent_dim   (сжатие)
  Декодер: latent_dim → 32 → 64 → input_dim    (разжатие)

Скрытые слои: ELU (совместимо с Unity Sentis)
Выходной слой: Tanh (для поддержки отрицательных значений — strains, iris)

Данные нормализуются в [-1..1] перед подачей в сеть.
"""

import torch
import torch.nn as nn


class FaceDenoiser(nn.Module):
    def __init__(self, input_dim: int = 20, latent_dim: int = 10):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, latent_dim),
            nn.LeakyReLU(0.01),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, input_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
