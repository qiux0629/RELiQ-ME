from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn


LEGACY_CENTER_NODE_FEATURES = [
    "swap_probability",
    "degree_ratio",
    "available_link_ratio",
    "mean_incident_fidelity",
    "decoupling_pulses_ratio",
    "is_terminal",
    "is_center",
    "distance_to_center_ratio",
    "is_virtual",
    "is_source_destination_valid",
]
LEGACY_ROUTING_DYNAMIC_FEATURES = [
    "is_current",
    "is_target",
    "is_visited",
    "distance_to_target_ratio",
]


@dataclass
class GraphBatch:
    """Torch tensors consumed by the quantum-network GNN encoder."""

    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    batch: Optional[torch.Tensor] = None

    @classmethod
    def from_feature_package(
        cls,
        feature_package: Dict[str, object],
        device: Optional[torch.device | str] = None,
    ) -> "GraphBatch":
        node_features = torch.as_tensor(
            np.asarray(feature_package["node_features"]),
            dtype=torch.float32,
            device=device,
        )
        edge_index = torch.as_tensor(
            np.asarray(feature_package["edge_index"]),
            dtype=torch.long,
            device=device,
        )
        edge_features = torch.as_tensor(
            np.asarray(feature_package["edge_features"]),
            dtype=torch.float32,
            device=device,
        )
        return cls(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
        )


def adapt_feature_package_dimensions(
    feature_package: Dict[str, object],
    node_feature_dim: Optional[int] = None,
    edge_feature_dim: Optional[int] = None,
) -> Dict[str, object]:
    """Trim/pad feature packages so old checkpoints remain evaluable."""

    package = dict(feature_package)
    node_features = np.asarray(package["node_features"], dtype=np.float32)
    edge_features = np.asarray(package["edge_features"], dtype=np.float32)

    if node_feature_dim is not None and node_features.shape[-1] != int(node_feature_dim):
        names = list(package.get("node_feature_names", []))
        preferred_names = LEGACY_CENTER_NODE_FEATURES
        if int(node_feature_dim) == len(LEGACY_CENTER_NODE_FEATURES) + len(LEGACY_ROUTING_DYNAMIC_FEATURES):
            preferred_names = LEGACY_CENTER_NODE_FEATURES + LEGACY_ROUTING_DYNAMIC_FEATURES
        if len(preferred_names) == int(node_feature_dim) and all(name in names for name in preferred_names):
            indices = [names.index(name) for name in preferred_names]
            node_features = node_features[:, indices]
            package["node_feature_names"] = preferred_names
        elif node_features.shape[-1] > int(node_feature_dim):
            node_features = node_features[:, : int(node_feature_dim)]
            package["node_feature_names"] = names[: int(node_feature_dim)]
        else:
            pad = np.zeros((node_features.shape[0], int(node_feature_dim) - node_features.shape[-1]), dtype=np.float32)
            node_features = np.concatenate([node_features, pad], axis=-1)
            package["node_feature_names"] = names + [f"padding_{i}" for i in range(pad.shape[-1])]

    if edge_feature_dim is not None and edge_features.shape[-1] != int(edge_feature_dim):
        names = list(package.get("edge_feature_names", []))
        if edge_features.shape[-1] > int(edge_feature_dim):
            edge_features = edge_features[:, : int(edge_feature_dim)]
            package["edge_feature_names"] = names[: int(edge_feature_dim)]
        else:
            pad = np.zeros((edge_features.shape[0], int(edge_feature_dim) - edge_features.shape[-1]), dtype=np.float32)
            edge_features = np.concatenate([edge_features, pad], axis=-1)
            package["edge_feature_names"] = names + [f"padding_{i}" for i in range(pad.shape[-1])]

    package["node_features"] = node_features
    package["edge_features"] = edge_features
    return package


@dataclass
class GNNEncoderOutput:
    node_embeddings: torch.Tensor
    graph_embedding: torch.Tensor
    batch: torch.Tensor


def _scatter_mean(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    output = values.new_zeros((dim_size, values.size(-1)))
    output.index_add_(0, index, values)

    counts = values.new_zeros((dim_size, 1))
    counts.index_add_(0, index, torch.ones((values.size(0), 1), device=values.device, dtype=values.dtype))
    return output / counts.clamp_min(1.0)


def _pool_nodes(node_embeddings: torch.Tensor, batch: torch.Tensor, pooling: str) -> torch.Tensor:
    graph_count = int(batch.max().item()) + 1 if batch.numel() else 1
    mean_pool = _scatter_mean(node_embeddings, batch, graph_count)

    if pooling == "mean":
        return mean_pool
    if pooling != "mean_max":
        raise ValueError(f"Unsupported pooling mode: {pooling}")

    max_pool = node_embeddings.new_full((graph_count, node_embeddings.size(-1)), -torch.inf)
    max_pool.scatter_reduce_(
        0,
        batch[:, None].expand_as(node_embeddings),
        node_embeddings,
        reduce="amax",
        include_self=True,
    )
    max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
    return torch.cat([mean_pool, max_pool], dim=-1)


class EdgeAwareMessagePassingLayer(nn.Module):
    """GraphSAGE-style message passing with edge-feature-conditioned messages."""

    def __init__(self, hidden_dim: int, edge_feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            aggregated = torch.zeros_like(node_embeddings)
        else:
            source, target = edge_index
            edge_context = self.edge_encoder(edge_features)
            messages = self.message_mlp(torch.cat([node_embeddings[source], edge_context], dim=-1))
            aggregated = _scatter_mean(messages, target, node_embeddings.size(0))

        updated = self.update_mlp(torch.cat([node_embeddings, aggregated], dim=-1))
        return self.norm(node_embeddings + updated)


class QuantumGraphGNNEncoder(nn.Module):
    """Standard GNN encoder for RELiQ quantum-network feature packages.

    The encoder performs multi-layer message passing over the full quantum network
    graph, uses physical edge attributes in every message, and returns both node
    embeddings and a graph-level embedding for downstream policy/value heads.
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 128,
        output_dim: Optional[int] = None,
        num_layers: int = 3,
        dropout: float = 0.1,
        pooling: str = "mean_max",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim or hidden_dim)
        self.pooling = pooling

        self.node_projection = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList(
            EdgeAwareMessagePassingLayer(hidden_dim, edge_feature_dim, dropout)
            for _ in range(num_layers)
        )
        pooled_dim = hidden_dim * 2 if pooling == "mean_max" else hidden_dim
        self.graph_projection = nn.Sequential(
            nn.Linear(pooled_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> GNNEncoderOutput:
        if batch is None:
            batch = torch.zeros(node_features.size(0), dtype=torch.long, device=node_features.device)

        node_embeddings = self.node_projection(node_features)
        for layer in self.layers:
            node_embeddings = layer(node_embeddings, edge_index, edge_features)

        graph_embedding = self.graph_projection(_pool_nodes(node_embeddings, batch, self.pooling))
        return GNNEncoderOutput(
            node_embeddings=node_embeddings,
            graph_embedding=graph_embedding,
            batch=batch,
        )

    @classmethod
    def from_feature_package(
        cls,
        feature_package: Dict[str, object],
        **kwargs,
    ) -> "QuantumGraphGNNEncoder":
        node_feature_dim = int(np.asarray(feature_package["node_features"]).shape[-1])
        edge_feature_dim = int(np.asarray(feature_package["edge_features"]).shape[-1])
        return cls(node_feature_dim=node_feature_dim, edge_feature_dim=edge_feature_dim, **kwargs)


class CenterSelectionHead(nn.Module):
    """Scores each node as a multipartite entanglement center candidate."""

    def __init__(self, hidden_dim: int, graph_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + graph_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        graph_embedding: torch.Tensor,
        batch: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        graph_context = graph_embedding[batch]
        scores = self.scorer(torch.cat([node_embeddings, graph_context], dim=-1)).squeeze(-1)
        if candidate_mask is not None:
            scores = scores.masked_fill(~candidate_mask.bool(), -torch.inf)
        return scores


def encode_feature_package(
    feature_package: Dict[str, object],
    encoder: Optional[QuantumGraphGNNEncoder] = None,
    device: Optional[torch.device | str] = None,
) -> GNNEncoderOutput:
    """Convenience helper for encoding one feature package with PyTorch tensors."""

    graph_batch = GraphBatch.from_feature_package(feature_package, device=device)
    encoder = encoder or QuantumGraphGNNEncoder.from_feature_package(feature_package).to(
        graph_batch.node_features.device
    )
    return encoder(
        graph_batch.node_features,
        graph_batch.edge_index,
        graph_batch.edge_features,
        batch=graph_batch.batch,
    )
