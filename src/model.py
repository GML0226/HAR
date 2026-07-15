from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class VisualEncoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(3, 24, 2), ConvBlock(24, 48, 2), ConvBlock(48, 96, 2), nn.AdaptiveAvgPool2d(1)
        )
        self.project = nn.Linear(96, embedding_dim)
        self.modality_embedding = nn.Parameter(torch.zeros(3, embedding_dim))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, modalities, frames, channels, height, width = images.shape
        frame_batch = images.reshape(batch * modalities * frames, channels, height, width).contiguous(memory_format=torch.channels_last)
        features = self.backbone(frame_batch).flatten(1)
        features = self.project(features).reshape(batch, modalities, frames, -1).mean(dim=2)
        return features + self.modality_embedding.unsqueeze(0)


class TemporalEncoder(nn.Module):
    def __init__(self, channels: int, embedding_dim: int) -> None:
        super().__init__()
        hidden = max(64, embedding_dim)
        self.features = nn.Sequential(
            nn.Conv1d(channels, hidden, 5, padding=2, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, embedding_dim, 1, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.features(sequence.transpose(1, 2)).mean(dim=-1)


class SixModalHAR(nn.Module):
    def __init__(self, embedding_dim: int = 128, dropout: float = 0.2, num_classes: int = 40) -> None:
        super().__init__()
        self.visual = VisualEncoder(embedding_dim)
        self.skeleton = TemporalEncoder(102, embedding_dim)
        self.imu = TemporalEncoder(12, embedding_dim)
        self.radar = TemporalEncoder(16, embedding_dim)
        self.gate = nn.Sequential(nn.Linear(embedding_dim, embedding_dim // 2), nn.ReLU(inplace=True), nn.Linear(embedding_dim // 2, 1))
        self.classifier = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Dropout(dropout), nn.Linear(embedding_dim, num_classes))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embeddings = torch.cat(
            [
                self.visual(batch["images"]),
                self.skeleton(batch["skeleton"]).unsqueeze(1),
                self.imu(batch["imu"]).unsqueeze(1),
                self.radar(batch["radar"]).unsqueeze(1),
            ],
            dim=1,
        )
        mask = batch["modality_mask"].bool()
        scores = self.gate(embeddings).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=1)
        fused = (weights.unsqueeze(-1) * embeddings).sum(dim=1)
        return {"logits": self.classifier(fused), "weights": weights}
