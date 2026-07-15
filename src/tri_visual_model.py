from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights, mobilenet_v3_small, resnet18

MODALITY_NAMES = ("Depth_Color", "IR", "Thermal")


class VisualBranch(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool, num_classes: int) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == "resnet18":
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.backbone.fc = nn.Identity()
            self.feature_dim = 512
        elif backbone_name == "mobilenet_v3_small":
            self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            self.backbone.classifier = nn.Identity()
            self.feature_dim = 576
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def set_trainable(self, stage: str) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = stage != "frozen"
        if stage != "last_stage":
            return
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if self.backbone_name == "resnet18":
            for parameter in self.backbone.layer4.parameters():
                parameter.requires_grad = True
        else:
            for block in self.backbone.features[-3:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True

    def forward(self, frames: torch.Tensor, quality: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, timesteps, channels, height, width = frames.shape
        image_batch = frames.reshape(batch * timesteps, channels, height, width).contiguous(memory_format=torch.channels_last)
        sequence = self.backbone(image_batch).reshape(batch, timesteps, self.feature_dim)
        pooled = sequence.mean(dim=1)
        return pooled, self.classifier(pooled)


class TriVisualGatedClassifier(nn.Module):
    def __init__(self, num_classes: int = 40, pretrained: bool = True, embedding_dim: int = 192, dropout: float = 0.2) -> None:
        super().__init__()
        self.branches = nn.ModuleDict({
            "Depth_Color": VisualBranch("resnet18", pretrained, num_classes),
            "IR": VisualBranch("mobilenet_v3_small", pretrained, num_classes),
            "Thermal": VisualBranch("mobilenet_v3_small", pretrained, num_classes),
        })
        self.projections = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(branch.feature_dim, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU())
            for name, branch in self.branches.items()
        })
        self.modality_embedding = nn.Parameter(torch.zeros(len(MODALITY_NAMES), embedding_dim))
        self.gate = nn.Sequential(nn.Linear(embedding_dim + 1, embedding_dim // 2), nn.GELU(), nn.Linear(embedding_dim // 2, 1))
        self.classifier = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Dropout(dropout), nn.Linear(embedding_dim, num_classes))

    def set_encoder_stage(self, stage: str) -> None:
        if stage not in {"frozen", "last_stage", "full"}:
            raise ValueError(f"Unsupported encoder stage: {stage}")
        for branch in self.branches.values():
            branch.set_trainable(stage)

    @staticmethod
    def apply_modality_dropout(mask: torch.Tensor, probability: float, training: bool) -> torch.Tensor:
        if not training or probability <= 0:
            return mask
        kept = mask & ~torch.rand_like(mask, dtype=torch.float32).lt(probability)
        empty = ~kept.any(dim=1)
        if empty.any():
            fallback = mask[empty].float().argmax(dim=1)
            kept[empty] = False
            kept[empty, fallback] = True
        return kept

    def forward(self, frames: torch.Tensor, modality_mask: torch.Tensor, quality: torch.Tensor, modality_dropout: float = 0.0) -> dict[str, torch.Tensor]:
        mask = self.apply_modality_dropout(modality_mask.bool(), modality_dropout, self.training)
        embeddings, auxiliary_logits = [], []
        for index, name in enumerate(MODALITY_NAMES):
            feature, auxiliary = self.branches[name](frames[:, index], quality[:, index])
            embeddings.append(self.projections[name](feature))
            auxiliary_logits.append(auxiliary)
        tokens = torch.stack(embeddings, dim=1) + self.modality_embedding.unsqueeze(0)
        gate_input = torch.cat([tokens, quality.unsqueeze(-1)], dim=-1)
        scores = self.gate(gate_input).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=1)
        fused = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        return {
            "logits": self.classifier(fused),
            "auxiliary_logits": torch.stack(auxiliary_logits, dim=1),
            "weights": weights,
            "effective_mask": mask,
        }
