"""
KG-XAI idea 3 (2026-10-11) -- does multi-task learning work on this trunk at
all, or was it specifically Phase 3's LVO/collateral labels (circular,
sparse, imbalanced) that made it collapse?

Deliberately NOT the Phase 4/5 per-vessel GNN design: that machinery exists
because LVO localization and collateral grading are LOCATION questions
(Phase 3's root-cause finding: global-average-pooling the whole bottleneck
throws away location, fatal for a location task). NIHSS at admission is a
whole-brain SEVERITY score, not a location question -- so a plain
global-average-pooled head on the segmentation bottleneck is the
architecturally right tool here, not a simplification that re-introduces
the old bug. If this collapses too, that's a real finding about this
trunk/data scale in general, not about pooling being the wrong choice for
this particular target.

Usage
-----
    from src.model.segresnet_aux import SegResNetWithAux
    model = SegResNetWithAux(in_channels=6)
    seg_logits, nihss_pred = model(x)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SegResNetWithAux(nn.Module):
    def __init__(self, in_channels: int = 6, init_filters: int = 16, dropout_prob: float = 0.2,
                 aux_hidden: int = 64):
        super().__init__()
        from monai.networks.nets import SegResNet
        self.trunk = SegResNet(spatial_dims=3, in_channels=in_channels, out_channels=2,
                                init_filters=init_filters, dropout_prob=dropout_prob)
        bottleneck_ch = init_filters * (2 ** (len(self.trunk.blocks_down) - 1))
        self.nihss_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(bottleneck_ch, aux_hidden), nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(aux_hidden, 1),
        )

    def forward(self, x: torch.Tensor):
        bottleneck, down_x = self.trunk.encode(x)
        # decode() expects down_x deepest-first; encode() appends
        # shallowest-first -- see segresnet_graph.py's _decode_with_graph,
        # which reverses it for exactly this reason before its own
        # (otherwise-equivalent) manual decode loop.
        seg_logits = self.trunk.decode(bottleneck, list(reversed(down_x)))
        nihss_pred = self.nihss_head(bottleneck).squeeze(-1)
        return seg_logits, nihss_pred

    def load_trunk_from_baseline(self, checkpoint_path: str, map_location=None):
        """Same warm-start as GraphInjectedSegResNet -- `.trunk` is an
        unmodified SegResNet, so a B0 checkpoint's keys match exactly. The
        new nihss_head starts random; segmentation output is unaffected
        until training moves it."""
        state = torch.load(checkpoint_path, map_location=map_location)
        self.trunk.load_state_dict(state)
