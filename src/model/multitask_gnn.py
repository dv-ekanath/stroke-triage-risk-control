"""
KG-XAI Phase 4/5 -- B1: the shared trunk (same as src/model/multitask.py's
B0), but the LVO/collateral heads now read from the vascular GNN
(src/model/gnn_head.py) over a whole-brain view instead of average-pooling
the segmentation-crop bottleneck. Segmentation itself is untouched -- same
encode()/decode() split, same trunk, so a warm-started B0 checkpoint loads
into `.trunk` directly.

Two forward passes per training step, because the two branches genuinely
need different views of the image (see build_transforms_gnn's docstring for
why the segmentation crop can miss the Circle of Willis entirely):

    forward_seg(image_patch)                 -> seg_logits
    forward_gnn(image_wb, node_coords,
                node_present, wb_size)        -> lvo_logits, col_logits

Usage
-----
    from src.model.multitask_gnn import MultiTaskGNN
    model = MultiTaskGNN(in_channels=6)
    seg_logits = model.forward_seg(patch)
    lvo_logits, col_logits = model.forward_gnn(wb_image, coords, present, wb_size)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.gnn_head import VesselGNNHead


class MultiTaskGNN(nn.Module):
    def __init__(self, in_channels: int = 6, n_lvo_classes: int = 4,
                 n_collateral_bins: int = 4, init_filters: int = 16,
                 dropout_prob: float = 0.2, gnn_stage: int = 1):
        super().__init__()
        from monai.networks.nets import SegResNet

        self.trunk = SegResNet(
            spatial_dims=3, in_channels=in_channels, out_channels=2,
            init_filters=init_filters, dropout_prob=dropout_prob,
        )
        # Which of encode()'s skip-connection feature maps to sample nodes
        # from. down_x[0] is full input resolution (convInit + stride-1
        # blocks, 16 channels); each later stage halves resolution and
        # doubles channels (down_x[1]: 1/2 res, 32ch -- down_x[3], the
        # bottleneck: 1/8 res, 128ch). Stage 1 is a deliberate middle
        # ground: half resolution still keeps R/L-MCA and R/L-ICA (a few cm
        # apart) resolvable as distinct sample points, while having passed
        # through one more conv block's receptive field than raw intensities.
        self.gnn_stage = gnn_stage
        channels_per_stage = [init_filters * (2 ** i) for i in range(len(self.trunk.blocks_down))]
        self.gnn_head = VesselGNNHead(
            feat_channels=channels_per_stage[gnn_stage],
            n_lvo_classes=n_lvo_classes, n_collateral_bins=n_collateral_bins,
        )

    def forward_seg(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, down_x = self.trunk.encode(x)
        down_x = list(down_x)
        down_x.reverse()
        return self.trunk.decode(bottleneck, down_x)

    def forward_gnn(self, x_wb: torch.Tensor, node_coords: torch.Tensor,
                     node_present: torch.Tensor, wb_size: tuple[int, int, int]):
        _, down_x = self.trunk.encode(x_wb)
        feat_map = down_x[self.gnn_stage]
        return self.gnn_head(feat_map, node_coords, node_present, wb_size)

    def load_trunk_from_baseline(self, checkpoint_path: str, map_location=None):
        """Same warm-start as MultiTaskSegResNet -- `.trunk` is an unmodified
        SegResNet, so a Review 1 (or B0) checkpoint's keys match exactly."""
        state = torch.load(checkpoint_path, map_location=map_location)
        self.trunk.load_state_dict(state)
