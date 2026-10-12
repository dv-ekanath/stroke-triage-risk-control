"""
KG-XAI Phase 5 -- B1(seg): inject the vascular-graph embeddings into the
segmentation decoder via cross-attention, instead of trying to predict LVO/
collateral from them (Phase 4/5's classification attempt -- SHELVED, see
CLAUDE.md: three training regimes all converged on a constant prediction).
This redirects the project's "does the knowledge graph help" claim onto a
task that has enough signal to test it fairly: does knowing the patient's
vascular anatomy (which vessels are present, how they connect) help the
model draw a better infarct outline?

Same shared trunk as B0/Phase 3/4 (`outputs/checkpoints/segresnet_fold0_best.pt`
loads directly into `.trunk`). Two forward passes, same reason as
Phase 4/5 (the segmentation crop can miss the Circle of Willis; the graph
branch needs the whole-brain view instead):

    node_emb = model.compute_node_embeddings(image_wb, node_coords, node_present, wb_size)
    seg_logits = model.forward_seg(image_patch, node_emb, node_present)

`forward_seg` reimplements SegResNet.decode()'s loop by hand (rather than
calling `self.trunk.decode()`) so a GraphCrossAttention module can run after
each up-sampling stage. It is architecturally IDENTICAL to
`self.trunk.decode()` when node_emb is None or GraphCrossAttention's gate is
at its zero-init value -- see GraphCrossAttention's docstring for why that
matters.

Usage
-----
    from src.model.segresnet_graph import GraphInjectedSegResNet
    model = GraphInjectedSegResNet(in_channels=6)
    node_emb = model.compute_node_embeddings(image_wb, node_coords, node_present, wb_size)
    seg_logits = model.forward_seg(image_patch, node_emb, node_present)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.gnn_head import VesselGNNEncoder


class GraphCrossAttention(nn.Module):
    """Decoder feature map (flattened voxels as queries) attends over the 12
    graph node embeddings (keys/values) -- cheap regardless of how many
    voxels there are, since there are only 12 keys. This is CONTENT-based
    attention, not spatially registered: it doesn't require knowing which
    voxel is physically closest to which vessel (the patch's random crop and
    the whole-brain grid aren't even in the same coordinate frame), it lets
    each voxel's features "ask" which vessel context is relevant to it.

    Zero-init residual gate: `self.gate` starts at exactly 0, so at the
    start of training this module is a mathematical no-op and the model is
    byte-identical to the plain warm-started trunk (verified in the smoke
    test). The graph pathway then has to EARN its way into affecting
    predictions via gradient descent, rather than immediately perturbing a
    trunk that already produces working segmentations -- standard practice
    for injecting a new pathway into a pretrained network (cf. adapter
    tuning), and important here specifically because we're warm-starting
    from a checkpoint we already know works.
    """

    def __init__(self, channels: int, graph_dim: int, heads: int = 4):
        super().__init__()
        assert channels % heads == 0, f"{channels=} not divisible by {heads=}"
        self.heads = heads
        self.head_dim = channels // heads
        self.q_proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.k_proj = nn.Linear(graph_dim, channels)
        self.v_proj = nn.Linear(graph_dim, channels)
        self.out_proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, node_emb: torch.Tensor, node_present: torch.Tensor) -> torch.Tensor:
        """x: [B, C, D, H, W] decoder features. node_emb: [B, 12, graph_dim].
        node_present: [B, 12]."""
        B, C, D, H, W = x.shape
        N = node_emb.shape[1]
        Hh, d = self.heads, self.head_dim

        q = self.q_proj(x).flatten(2).transpose(1, 2)          # [B, D*H*W, C]
        k = self.k_proj(node_emb)                                # [B, N, C]
        v = self.v_proj(node_emb)                                # [B, N, C]

        q = q.view(B, -1, Hh, d).transpose(1, 2)                 # [B, Hh, DHW, d]
        k = k.view(B, N, Hh, d).transpose(1, 2)                  # [B, Hh, N, d]
        v = v.view(B, N, Hh, d).transpose(1, 2)                  # [B, Hh, N, d]

        attn = (q @ k.transpose(-1, -2)) / (d ** 0.5)             # [B, Hh, DHW, N]
        mask = (node_present == 0).view(B, 1, 1, N)
        attn = attn.masked_fill(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        # a sample with ZERO present nodes -> softmax over all -inf -> NaN;
        # zero it out rather than let NaN reach the residual add
        attn = torch.nan_to_num(attn, nan=0.0)

        out = attn @ v                                            # [B, Hh, DHW, d]
        out = out.transpose(1, 2).reshape(B, D * H * W, C).transpose(1, 2).view(B, C, D, H, W)
        out = self.out_proj(out)
        return x + self.gate * out


class GraphInjectedSegResNet(nn.Module):
    def __init__(self, in_channels: int = 6, init_filters: int = 16, dropout_prob: float = 0.2,
                 gnn_stage: int = 1, graph_hidden: int = 32, attn_heads: int = 4,
                 edge_index: torch.Tensor | None = None):
        super().__init__()
        from monai.networks.nets import SegResNet

        self.trunk = SegResNet(
            spatial_dims=3, in_channels=in_channels, out_channels=2,
            init_filters=init_filters, dropout_prob=dropout_prob,
        )
        self.gnn_stage = gnn_stage
        channels_per_stage = [init_filters * (2 ** i) for i in range(len(self.trunk.blocks_down))]
        # edge_index override: the random-edge-GNN ablation (CLAUDE.md,
        # 2026-10) swaps the real CoW anatomy for a same-size random graph;
        # default (None) is the real anatomy, unchanged from Phase 5/6/8.
        self.vessel_gnn = VesselGNNEncoder(feat_channels=channels_per_stage[gnn_stage], hidden=graph_hidden,
                                            edge_index=edge_index)

        # one GraphCrossAttention per decode stage, channel-matched to that
        # stage's feature map (see SegResNet._make_up_layers: channels halve
        # each stage going up from the bottleneck)
        n_up = len(self.trunk.up_layers)
        decode_channels = [init_filters * (2 ** i) for i in range(n_up - 1, -1, -1)]
        self.cross_attn = nn.ModuleList([
            GraphCrossAttention(c, graph_hidden, heads=attn_heads) for c in decode_channels
        ])

    def compute_node_embeddings(self, x_wb: torch.Tensor, node_coords: torch.Tensor,
                                 node_present: torch.Tensor, wb_size: tuple[int, int, int]) -> torch.Tensor:
        _, down_x = self.trunk.encode(x_wb)
        feat_map = down_x[self.gnn_stage]
        return self.vessel_gnn(feat_map, node_coords, node_present, wb_size)

    def _decode_with_graph(self, bottleneck: torch.Tensor, down_x: list[torch.Tensor],
                            node_emb: torch.Tensor, node_present: torch.Tensor) -> torch.Tensor:
        down_x = list(down_x)
        down_x.reverse()
        x = bottleneck
        for i, (up, upl, attn) in enumerate(zip(self.trunk.up_samples, self.trunk.up_layers, self.cross_attn)):
            x = up(x) + down_x[i + 1]
            x = upl(x)
            x = attn(x, node_emb, node_present)
        if self.trunk.use_conv_final:
            x = self.trunk.conv_final(x)
        return x

    def forward_wb_seg_and_graph(self, x_wb: torch.Tensor, node_coords: torch.Tensor,
                                  node_present: torch.Tensor, wb_size: tuple[int, int, int]):
        """Phase 6: one encode(x_wb) pass, reused for BOTH the node
        embeddings (as compute_node_embeddings does) AND a whole-brain
        segmentation prediction -- needed for the laterality consistency
        loss, which has to compare predicted-lesion mass against the
        midline over the SAME fixed, unaugmented wb grid that
        `midline_voxel_x` (src/model/dataset.py:ExtractVesselCoordsd) is
        computed in. Calling compute_node_embeddings() + forward_seg()
        separately would encode(x_wb) twice for no reason; this shares it.

        Returns (wb_seg_logits, node_emb) -- node_emb can be reused for the
        patch branch's forward_seg() call in the same training step."""
        bottleneck, down_x = self.trunk.encode(x_wb)
        feat_map = down_x[self.gnn_stage]
        node_emb = self.vessel_gnn(feat_map, node_coords, node_present, wb_size)
        wb_seg_logits = self._decode_with_graph(bottleneck, down_x, node_emb, node_present)
        return wb_seg_logits, node_emb

    def forward_seg(self, x_patch: torch.Tensor, node_emb: torch.Tensor,
                     node_present: torch.Tensor) -> torch.Tensor:
        """Reimplements SegResNet.decode()'s loop by hand (via
        _decode_with_graph) so GraphCrossAttention can run after each
        up-sampling stage -- see this module's docstring."""
        bottleneck, down_x = self.trunk.encode(x_patch)
        return self._decode_with_graph(bottleneck, down_x, node_emb, node_present)

    def forward(self, x_patch: torch.Tensor, x_wb: torch.Tensor, node_coords: torch.Tensor,
                node_present: torch.Tensor, wb_size: tuple[int, int, int]) -> torch.Tensor:
        """Convenience one-call form for the training loop. Validation calls
        compute_node_embeddings() once per subject and forward_seg() many
        times (once per sliding-window patch) instead -- see train_graph_seg.py."""
        node_emb = self.compute_node_embeddings(x_wb, node_coords, node_present, wb_size)
        return self.forward_seg(x_patch, node_emb, node_present)

    def load_trunk_from_baseline(self, checkpoint_path: str, map_location=None):
        """Same warm-start as MultiTaskSegResNet/MultiTaskGNN -- `.trunk` is
        an unmodified SegResNet, so a Review 1/B0 checkpoint's keys match
        exactly. The graph pathway (vessel_gnn, cross_attn) starts from
        random init except each GraphCrossAttention's zero-init gate, so the
        model's segmentation output is byte-identical to the checkpoint's
        own until training moves the gates away from zero."""
        state = torch.load(checkpoint_path, map_location=map_location)
        self.trunk.load_state_dict(state)
