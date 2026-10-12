"""
KG-XAI Phase 4/5 -- the vascular-graph head: samples the shared encoder's
feature map at each of the 12 Circle-of-Willis nodes' real locations, runs a
GNN over the fixed anatomical edges (kg/cow_topology.json), and predicts LVO
localization + collateral proxy from the result.

This replaces Phase 3's failed design (global-average-pool the whole
bottleneck into one vector, per src/model/multitask.py) -- that threw away
location entirely, which is fatal for a task that IS a location question.
Here each node gets ITS OWN feature vector sampled from where that vessel
actually is, and the graph's edges (ICA->MCA, the Pcom/Acom collateral
routes) let the GNN move information between vessels the way blood actually
would.

Leakage note: node_coords/node_present come from the real cow-msk (ses-01,
admission data) -- allowed per kg/cow_topology.json's leakage_rule. They are
NOT the same thing as the `occluded`/`nearest_vessel` features in
kg_node_features.json, which are computed FROM the LVO mask and would make
predicting lvo_class circular -- this module never touches those.

Usage
-----
    from src.model.gnn_head import VesselGNNHead
    head = VesselGNNHead(feat_channels=64, n_lvo_classes=4, n_collateral_bins=4)
    lvo_logits, col_logits = head(feat_map, node_coords, node_present, wb_size)
"""

from __future__ import annotations

import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.graph.build import edge_index, load_graph

KG_DIR = pathlib.Path("kg")


def cow_edge_index() -> torch.Tensor:
    """The fixed 12-node Circle-of-Willis graph, as a torch LongTensor
    edge_index [2, 24] (both directions). Same file every subject uses --
    anatomical variants are absent NODES (node_present=0), never different
    edges, per cow_topology.json's own description."""
    graph = load_graph(KG_DIR / "cow_topology.json")
    ei = edge_index(graph)  # numpy [2, n_edges*2], see src/graph/build.py
    return torch.from_numpy(ei).long()


def random_edge_index(seed: int = 0, n_nodes: int = 12, n_edges: int = 12) -> torch.Tensor:
    """A random graph over the same 12 nodes, same edge COUNT as the real
    anatomy (12 undirected edges, cow_topology.json), same bidirectional
    edge_index format -- so the GAT layers see the same amount of
    connectivity, and the only thing that differs from cow_edge_index() is
    WHICH nodes talk to which. For the "is it the real anatomy, or just
    having a graph at all" ablation (CLAUDE.md, 2026-10).

    Deterministic and side-effect-free: uses a local random.Random(seed),
    not the global RNG, so calling this twice with the same seed (once at
    train time, once at eval time) reproduces the identical graph without
    needing to store it in the checkpoint -- edge_index is a
    persistent=False buffer (see VesselGNNEncoder), so it is NOT saved in
    state_dict() by design."""
    import random
    rng = random.Random(seed)
    all_pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    chosen = rng.sample(all_pairs, n_edges)
    src = [a for a, b in chosen] + [b for a, b in chosen]
    dst = [b for a, b in chosen] + [a for a, b in chosen]
    return torch.tensor([src, dst], dtype=torch.long)


def node_order() -> list[str]:
    graph = load_graph(KG_DIR / "cow_topology.json")
    return [n["id"] for n in graph["nodes"]]


def topcow_label_order() -> list[int]:
    graph = load_graph(KG_DIR / "cow_topology.json")
    return [n["topcow_label"] for n in graph["nodes"]]


def sample_node_features(feat_map: torch.Tensor, node_coords: torch.Tensor,
                          wb_size: tuple[int, int, int]) -> torch.Tensor:
    """feat_map: [B, C, S0, S1, S2] (any resolution -- the encoder's
    downsampled feature map, e.g. from trunk.encode()'s bottleneck or one of
    the skip connections). node_coords: [B, N, 3] voxel indices INTO THE
    WB_SIZE GRID (i.e. the resolution the image tensor going into the
    encoder had, from ExtractVesselCoordsd), not into feat_map's own
    resolution -- normalized coordinates are resolution-independent, so this
    works regardless of feat_map's downsampling stride.

    Returns [B, N, C] -- one feature vector per node, trilinearly sampled.
    """
    B, C = feat_map.shape[0], feat_map.shape[1]
    N = node_coords.shape[1]
    wb = torch.tensor(wb_size, dtype=node_coords.dtype, device=node_coords.device)
    # normalize to [-1, 1] per axis; size-1 so index 0 -> -1 and size-1 -> 1
    norm = 2.0 * node_coords / (wb - 1).clamp(min=1) - 1.0  # [B, N, 3], order (S0,S1,S2)

    # grid_sample on a 5D input expects the grid's last dim as (x,y,z) i.e.
    # indexing (W,H,D) = reverse of our (S0,S1,S2) array-index order.
    grid = norm.flip(-1).view(B, N, 1, 1, 3)
    sampled = F.grid_sample(feat_map, grid, mode="bilinear",
                             padding_mode="zeros", align_corners=True)  # [B, C, N, 1, 1]
    return sampled.view(B, C, N).permute(0, 2, 1)  # [B, N, C]


class _SimpleGATLayer(nn.Module):
    """A minimal batched multi-head graph-attention layer over a FIXED, tiny
    graph (12 nodes, 24 directed edges) shared by every sample -- small
    enough that hand-rolling the batching (rather than torch_geometric's
    Batch machinery) is simpler to keep inside a plain MONAI/PyTorch training
    loop. torch_geometric's GATConv itself does the actual attention math.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 2):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.conv = GATConv(in_dim, out_dim, heads=heads, concat=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """x: [B, N, C_in]. edge_index: [2, E] for ONE graph (0..N-1).
        Batches by offsetting node indices per sample into one big graph,
        the standard PyG pattern, done by hand here to avoid a Data/Batch
        round-trip through the dict-based MONAI pipeline."""
        B, N, C = x.shape
        E = edge_index.shape[1]
        offsets = (torch.arange(B, device=x.device) * N).view(B, 1, 1)
        batched_edges = (edge_index.view(1, 2, E) + offsets).permute(1, 0, 2).reshape(2, B * E)
        flat_x = x.reshape(B * N, C)
        out = self.conv(flat_x, batched_edges)
        return out.view(B, N, -1)


class VesselGNNEncoder(nn.Module):
    """Samples the encoder's feature map at each of the 12 real vessel
    locations and runs 2 GATConv layers over the fixed anatomical edges,
    returning PER-NODE embeddings -- no pooling, no task head. Shared by
    VesselGNNHead (Phase 4/5's classification attempt, shelved -- see
    CLAUDE.md) and Phase 5's GraphCrossAttention (src/model/segresnet_graph.py),
    which injects these same per-node embeddings into the segmentation
    decoder instead of pooling them into one vector."""

    def __init__(self, feat_channels: int, hidden: int = 32, dropout: float = 0.2,
                 edge_index: torch.Tensor | None = None):
        super().__init__()
        # override lets the random-edge ablation (random_edge_index above)
        # swap in a same-size random graph; default is the real anatomy
        self.register_buffer("edge_index", edge_index if edge_index is not None else cow_edge_index(),
                              persistent=False)
        self.hidden = hidden
        # +1 for the node_present flag, concatenated after sampling
        self.node_proj = nn.Linear(feat_channels + 1, hidden)
        self.gat1 = _SimpleGATLayer(hidden, hidden)
        self.gat2 = _SimpleGATLayer(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feat_map: torch.Tensor, node_coords: torch.Tensor,
                node_present: torch.Tensor, wb_size: tuple[int, int, int]) -> torch.Tensor:
        """Returns [B, 12, hidden] -- one embedding per vessel node."""
        feats = sample_node_features(feat_map, node_coords, wb_size)  # [B, N, C]
        feats = feats * node_present.unsqueeze(-1)  # zero out absent vessels' sampled content
        x = torch.cat([feats, node_present.unsqueeze(-1)], dim=-1)  # [B, N, C+1]

        x = F.elu(self.node_proj(x))
        x = self.dropout(x)
        x = F.elu(self.gat1(x, self.edge_index))
        x = self.dropout(x)
        x = self.gat2(x, self.edge_index)  # [B, N, hidden]
        return x


def masked_mean_max_pool(x: torch.Tensor, node_present: torch.Tensor) -> torch.Tensor:
    """[B, N, C] node embeddings + [B, N] present mask -> [B, 2C] graph-level
    readout (mean and max over present nodes only, concatenated). A sample
    with zero present nodes gets an all-zero readout rather than -inf/NaN."""
    present = node_present.unsqueeze(-1)  # [B, N, 1]
    n_present = present.sum(dim=1).clamp(min=1.0)
    mean_pool = (x * present).sum(dim=1) / n_present
    masked_for_max = x.masked_fill(present == 0, float("-inf"))
    max_pool = masked_for_max.amax(dim=1)  # [B, C]; -inf per-channel only if ALL nodes absent
    max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
    any_present = (n_present > 0).float()
    return torch.cat([mean_pool, max_pool], dim=-1) * any_present


class VesselGNNHead(nn.Module):
    """Phase 4/5's classification attempt (LVO localization + collateral
    proxy from a pooled graph readout) -- SHELVED per CLAUDE.md: three
    independent training regimes (plain, augmented+reweighted, frozen-trunk)
    all converged on predicting one constant class for every subject. Kept
    here, unused, in case it's revisited later; current work uses
    VesselGNNEncoder directly (src/model/segresnet_graph.py)."""

    def __init__(self, feat_channels: int, n_lvo_classes: int = 4,
                 n_collateral_bins: int = 4, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.encoder = VesselGNNEncoder(feat_channels, hidden=hidden, dropout=dropout)
        readout_dim = hidden * 2  # mean + max pool over nodes
        self.lvo_head = nn.Sequential(
            nn.Linear(readout_dim, 32), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(32, n_lvo_classes),
        )
        self.collateral_head = nn.Sequential(
            nn.Linear(readout_dim, 32), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(32, n_collateral_bins),
        )

    def forward(self, feat_map: torch.Tensor, node_coords: torch.Tensor,
                node_present: torch.Tensor, wb_size: tuple[int, int, int]):
        node_emb = self.encoder(feat_map, node_coords, node_present, wb_size)
        readout = masked_mean_max_pool(node_emb, node_present)
        return self.lvo_head(readout), self.collateral_head(readout)
