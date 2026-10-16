"""Parallel multikernel CNN plus Transformer for dense heartbeat labeling."""

import torch
from torch import nn
import torch.nn.functional as F

from .label_config import NUM_CLASSES, WINDOW_SAMPLES


class DownsampleBranch(nn.Module):
    """Five stride-2 convolutions with one receptive-field size per branch."""

    def __init__(self, kernel_size: int, channels=(8, 16, 32, 64, 128)):
        super().__init__()
        layers = []
        in_channels = 1
        for out_channels in channels:
            layers.extend([
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    bias=False,
                ),
                nn.GroupNorm(8, out_channels),
                nn.GELU(),
            ])
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(8, out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        return self.activation(self.norm(self.conv(x)))


class MultiScaleCNNTransformer(nn.Module):
    """Three kernel-size branches, Transformer context, and dense 6-class output."""

    def __init__(
        self,
        input_samples: int = WINDOW_SAMPLES,
        num_classes: int = NUM_CLASSES,
        model_dim: int = 192,
        attention_heads: int = 6,
        transformer_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_samples % 32 != 0:
            raise ValueError("Input length must be divisible by 32.")
        if model_dim % attention_heads != 0:
            raise ValueError("model_dim must be divisible by attention_heads.")

        self.input_samples = input_samples
        self.branches = nn.ModuleList([
            DownsampleBranch(kernel_size=3),
            DownsampleBranch(kernel_size=5),
            DownsampleBranch(kernel_size=7),
        ])
        self.fusion = nn.Sequential(
            nn.Conv1d(3 * 128, model_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, model_dim),
            nn.GELU(),
        )

        token_count = input_samples // 32
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=attention_heads,
            dim_feedforward=model_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers, enable_nested_tensor=False
        )
        self.position = nn.Parameter(torch.zeros(1, token_count, model_dim))
        nn.init.trunc_normal_(self.position, std=0.02)

        self.decoder = nn.Sequential(
            UpBlock(model_dim, 128),
            UpBlock(128, 96),
            UpBlock(96, 64),
            UpBlock(64, 32),
            UpBlock(32, 32),
        )
        self.classifier = nn.Conv1d(32, num_classes, kernel_size=1)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"Expected input [batch, 1, samples], got {tuple(x.shape)}")
        if x.shape[-1] != self.input_samples:
            raise ValueError(
                f"Expected {self.input_samples} samples, got {x.shape[-1]}"
            )

        branch_features = [branch(x) for branch in self.branches]
        fused = self.fusion(torch.cat(branch_features, dim=1))
        tokens = fused.transpose(1, 2)
        tokens = self.transformer(tokens + self.position)
        features = tokens.transpose(1, 2)
        decoded = self.decoder(features)
        return self.classifier(decoded)