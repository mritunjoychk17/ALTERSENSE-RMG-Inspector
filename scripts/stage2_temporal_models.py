#!/usr/bin/env python3
"""Shared temporal Stage 2 model components."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models


POSE_LABELS = ["align", "get", "idle", "put", "sew", "unknown"]


def build_backbone() -> tuple[nn.Module, int]:
    backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    return backbone.features, 576


def pose_feature_dim() -> int:
    return len(POSE_LABELS) + 1


def encode_pose_label(label: str) -> list[float]:
    label = (label or "").strip().lower()
    out = [0.0] * len(POSE_LABELS)
    try:
        idx = POSE_LABELS.index(label)
    except ValueError:
        idx = POSE_LABELS.index("unknown")
    out[idx] = 1.0
    return out


class TemporalConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class FrameEncoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.encoder, feat_dim = build_backbone()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.frame_proj = nn.Linear(feat_dim, hidden_dim)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = clips.shape
        x = clips.view(b * t, c, h, w)
        feats = self.encoder(x)
        feats = self.pool(feats).flatten(1)
        feats = self.frame_proj(feats)
        return feats.view(b, t, -1)


class ClipGRUModel(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.frame = FrameEncoder(hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, clips: torch.Tensor, pose_features: torch.Tensor | None = None) -> torch.Tensor:
        feats = self.frame(clips)
        _, hidden = self.gru(feats)
        return self.head(hidden[-1])


class ClipTCNModel(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 256, depth: int = 4) -> None:
        super().__init__()
        self.frame = FrameEncoder(hidden_dim)
        self.tcn = nn.Sequential(*[TemporalConvBlock(hidden_dim, dilation=2**idx) for idx in range(depth)])
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, clips: torch.Tensor, pose_features: torch.Tensor | None = None) -> torch.Tensor:
        feats = self.frame(clips).transpose(1, 2)
        feats = self.tcn(feats)
        pooled = feats.mean(dim=2)
        return self.head(pooled)


class ClipPoseHybridModel(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 256, depth: int = 4, pose_dim: int | None = None) -> None:
        super().__init__()
        pose_dim = pose_dim or pose_feature_dim()
        self.frame = FrameEncoder(hidden_dim)
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.tcn = nn.Sequential(*[TemporalConvBlock(hidden_dim * 2, dilation=2**idx) for idx in range(depth)])
        self.head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, clips: torch.Tensor, pose_features: torch.Tensor | None = None) -> torch.Tensor:
        if pose_features is None:
            raise ValueError("Hybrid pose model requires pose_features.")
        frame_feats = self.frame(clips)
        pose_feats = self.pose_proj(pose_features)
        feats = torch.cat([frame_feats, pose_feats], dim=2).transpose(1, 2)
        feats = self.tcn(feats)
        pooled = feats.mean(dim=2)
        return self.head(pooled)


class Clip3DCNNModel(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 256) -> None:
        super().__init__()
        mid_channels = max(hidden_dim // 4, 32)
        high_channels = max(hidden_dim // 2, 64)
        self.net = nn.Sequential(
            nn.Conv3d(3, mid_channels, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(mid_channels, high_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(high_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(high_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, clips: torch.Tensor, pose_features: torch.Tensor | None = None) -> torch.Tensor:
        x = clips.transpose(1, 2)
        feats = self.net(x)
        pooled = self.pool(feats)
        return self.head(pooled)


def build_model(model_type: str, num_classes: int, hidden_dim: int = 256) -> nn.Module:
    model_type = model_type.strip().lower()
    if model_type == "gru":
        return ClipGRUModel(num_classes=num_classes, hidden_dim=hidden_dim)
    if model_type == "tcn":
        return ClipTCNModel(num_classes=num_classes, hidden_dim=hidden_dim)
    if model_type == "hybrid_pose":
        return ClipPoseHybridModel(num_classes=num_classes, hidden_dim=hidden_dim)
    if model_type == "cnn3d":
        return Clip3DCNNModel(num_classes=num_classes, hidden_dim=hidden_dim)
    raise ValueError(f"Unsupported model_type: {model_type}")
