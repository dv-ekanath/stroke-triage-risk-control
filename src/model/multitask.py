"""
KG-XAI Phase 3 -- extend the Review 1 baseline with two new heads (H2 LVO
localization, H3 collateral proxy), reusing the trained segmentation trunk
rather than rebuilding the deck's 3-encoder architecture from scratch (see
PROJECT_PLAN.md Sec 6, "Multimodal encoder & fusion" -- a disclosed
simplification, given the 2-week window).

Why this is a real shared trunk, not a hack: MONAI's SegResNet already
splits its own forward() into public encode()/decode() methods --
    def forward(self, x):
        x, down_x = self.encode(x)
        down_x.reverse()
        return self.decode(x, down_x)
so calling the same two methods here reproduces the ORIGINAL model's
segmentation computation exactly, then taps the bottleneck feature `x`
(before decode) for the two new heads. This is not a hook or a monkeypatch,
it's the same public API SegResNet's own forward() uses -- zero risk of
silently changing segmentation behaviour, and existing Review 1 checkpoints
(outputs/checkpoints/segresnet_fold0_best.pt) load directly into the
`.trunk` submodule since it's an unmodified SegResNet.

Usage
-----
    from src.model.multitask import MultiTaskSegResNet
    model = MultiTaskSegResNet(in_channels=6)
    seg_logits, lvo_logits, collateral_logits = model(image)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskSegResNet(nn.Module):
    def __init__(self, in_channels: int = 6, n_lvo_classes: int = 4,
                 n_collateral_bins: int = 4, init_filters: int = 16,
                 dropout_prob: float = 0.2):
        super().__init__()
        from monai.networks.nets import SegResNet

        self.trunk = SegResNet(
            spatial_dims=3, in_channels=in_channels, out_channels=2,
            init_filters=init_filters, dropout_prob=dropout_prob,
        )
        # Bottleneck channel count: SegResNet's default blocks_down has 4
        # stages, channels double each stage from init_filters.
        bottleneck_channels = init_filters * (2 ** (len(self.trunk.blocks_down) - 1))

        self.lvo_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(bottleneck_channels, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(64, n_lvo_classes),
        )
        self.collateral_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(bottleneck_channels, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(64, n_collateral_bins),
        )

    def forward(self, x: torch.Tensor):
        bottleneck, down_x = self.trunk.encode(x)
        lvo_logits = self.lvo_head(bottleneck)
        collateral_logits = self.collateral_head(bottleneck)

        down_x = list(down_x)
        down_x.reverse()
        seg_logits = self.trunk.decode(bottleneck, down_x)

        return seg_logits, lvo_logits, collateral_logits

    def load_trunk_from_baseline(self, checkpoint_path: str, map_location=None):
        """Warm-start the shared trunk from the Review 1 baseline checkpoint
        (plain SegResNet state_dict -- keys match exactly since .trunk is an
        unmodified SegResNet). The two new heads start from random init."""
        state = torch.load(checkpoint_path, map_location=map_location)
        self.trunk.load_state_dict(state)
