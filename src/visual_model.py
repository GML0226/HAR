from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18MeanPool(nn.Module):
    def __init__(self, num_classes: int = 40, pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, num_classes))

    def set_backbone_trainable(self, enabled: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = enabled

    def forward(self, frames: torch.Tensor, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, timesteps, channels, height, width = frames.shape
        frame_batch = frames.reshape(batch * timesteps, channels, height, width).contiguous(memory_format=torch.channels_last)
        features = self.backbone(frame_batch).reshape(batch, timesteps, -1)
        if frame_mask is None:
            pooled = features.mean(dim=1)
        else:
            weights = frame_mask.view(batch, 1, 1).expand(-1, timesteps, 1)
            pooled = (features * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled)
