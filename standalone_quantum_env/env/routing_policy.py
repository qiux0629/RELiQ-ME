from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from env.gnn_encoder import GNNEncoderOutput, QuantumGraphGNNEncoder
from env.multipartite import _edge_available_links, build_gnn_feature_package


@dataclass
class RoutingPolicyOutput:
    logits: torch.Tensor
    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    gnn_output: GNNEncoderOutput


def build_bipartite_feature_package(
    network,
    current: int,
    target: int,
    visited: Optional[Sequence[int]] = None,
) -> dict:
    """Build graph features for one step of a bipartite terminal-to-target route."""

    visited_set = set(int(node) for node in (visited or ()))
    current = int(current)
    target = int(target)
    package = build_gnn_feature_package(network, terminals=[current, target], center=target)
    node_features = np.asarray(package["node_features"], dtype=np.float32)
    dynamic_features = []
    for node in network.nodes:
        distance_to_target = 0.0
        if node.id in network.shortest_paths_weights and target in network.shortest_paths_weights[node.id]:
            distance_to_target = network.shortest_paths_weights[node.id][target] / max(network.diameter, 1)
        dynamic_features.append(
            [
                1.0 if node.id == current else 0.0,
                1.0 if node.id == target else 0.0,
                1.0 if node.id in visited_set else 0.0,
                float(distance_to_target),
            ]
        )
    package["node_features"] = np.concatenate(
        [node_features, np.asarray(dynamic_features, dtype=np.float32)],
        axis=-1,
    )
    package["node_feature_names"] = list(package["node_feature_names"]) + [
        "is_current",
        "is_target",
        "is_visited",
        "distance_to_target_ratio",
    ]
    return package


def action_neighbor_nodes(network, current: int, device: Optional[torch.device | str] = None) -> torch.Tensor:
    """Return node ids represented by action slots; slot 0 is idle and uses -1."""

    current_node = network.get_nodes(0)[int(current)]
    action_nodes = [-1]
    for slot in range(network.neighbor_count):
        if slot < len(current_node.edges):
            edge = network.edges[current_node.edges[slot]]
            action_nodes.append(int(edge.get_other_node(int(current))))
        else:
            action_nodes.append(-1)
    return torch.as_tensor(action_nodes, dtype=torch.long, device=device)


def action_for_next_node(network, current: int, next_node: int) -> Optional[int]:
    current_node = network.get_nodes(0)[int(current)]
    for slot, edge_id in enumerate(current_node.edges[: network.neighbor_count]):
        if network.edges[edge_id].get_other_node(int(current)) == int(next_node):
            return slot + 1
    return None


def valid_routing_action_mask(
    network,
    current: int,
    target: int,
    visited: Optional[Sequence[int]] = None,
    device: Optional[torch.device | str] = None,
    allow_idle: bool = False,
    prefer_progress: bool = False,
) -> torch.Tensor:
    """Return a True-for-valid action mask for next-hop routing."""

    visited_set = set(int(node) for node in (visited or ()))
    mask = torch.zeros(network.neighbor_count + 1, dtype=torch.bool, device=device)
    if int(current) == int(target):
        mask[0] = True
        return mask

    current_distance = None
    if (
        prefer_progress
        and int(current) in network.shortest_paths_weights
        and int(target) in network.shortest_paths_weights[int(current)]
    ):
        current_distance = network.shortest_paths_weights[int(current)][int(target)]

    current_node = network.get_nodes(0)[int(current)]
    progress_actions = []
    for slot, edge_id in enumerate(current_node.edges[: network.neighbor_count]):
        edge = network.edges[edge_id]
        other = int(edge.get_other_node(int(current)))
        if other in visited_set:
            continue
        if _edge_available_links(edge) <= 0:
            continue
        if not nx.has_path(network.G, other, int(target)):
            continue
        mask[slot + 1] = True
        if (
            current_distance is not None
            and other in network.shortest_paths_weights
            and int(target) in network.shortest_paths_weights[other]
            and network.shortest_paths_weights[other][int(target)] < current_distance
        ):
            progress_actions.append(slot + 1)

    if prefer_progress and progress_actions:
        progress_mask = torch.zeros_like(mask)
        progress_mask[torch.as_tensor(progress_actions, dtype=torch.long, device=device)] = True
        mask = progress_mask

    if not bool(mask[1:].any()):
        mask[0] = True
    elif allow_idle:
        # Waiting is only legal when routing is blocked. This prevents the
        # policy from learning that idle is an attractive early-stop action.
        mask[0] = False
    return mask


class NextHopActionHead(nn.Module):
    """Scores idle plus neighbor-edge action slots for one routing request."""

    def __init__(self, hidden_dim: int, graph_dim: int, max_actions: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.max_actions = int(max_actions)
        self.idle_embedding = nn.Parameter(torch.zeros(hidden_dim))
        self.slot_embedding = nn.Embedding(self.max_actions, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4 + graph_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        graph_embedding: torch.Tensor,
        batch: torch.Tensor,
        current_node: int,
        target_node: int,
        action_neighbor_ids: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        del batch
        current_embedding = node_embeddings[int(current_node)]
        target_embedding = node_embeddings[int(target_node)]
        graph_context = graph_embedding[0]
        action_representations = []
        for action, neighbor_id in enumerate(action_neighbor_ids.tolist()):
            if neighbor_id < 0:
                neighbor_embedding = self.idle_embedding
            else:
                neighbor_embedding = node_embeddings[int(neighbor_id)]
            slot_embedding = self.slot_embedding(
                torch.as_tensor(action, dtype=torch.long, device=node_embeddings.device)
            )
            action_representations.append(
                torch.cat(
                    [
                        current_embedding,
                        target_embedding,
                        neighbor_embedding,
                        slot_embedding,
                        graph_context,
                    ],
                    dim=-1,
                )
            )
        logits = self.scorer(torch.stack(action_representations, dim=0)).squeeze(-1)
        return logits.masked_fill(~action_mask.bool(), -torch.inf)


class RoutingValueHead(nn.Module):
    """State-value head for actor-critic next-hop routing."""

    def __init__(self, hidden_dim: int, graph_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(hidden_dim * 2 + graph_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        graph_embedding: torch.Tensor,
        current_node: int,
        target_node: int,
    ) -> torch.Tensor:
        return self.value(
            torch.cat(
                [
                    node_embeddings[int(current_node)],
                    node_embeddings[int(target_node)],
                    graph_embedding[0],
                ],
                dim=-1,
            )
        ).squeeze(-1)


class BipartiteRoutingPolicy(nn.Module):
    """GNN policy that chooses the next hop for a bipartite entanglement route."""

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        max_actions: int,
        hidden_dim: int = 128,
        graph_output_dim: int = 128,
        num_gnn_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gnn = QuantumGraphGNNEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=graph_output_dim,
            num_layers=num_gnn_layers,
            dropout=dropout,
        )
        self.action_head = NextHopActionHead(hidden_dim, graph_output_dim, max_actions, dropout)
        self.value_head = RoutingValueHead(hidden_dim, graph_output_dim, dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        current_node: int,
        target_node: int,
        action_neighbor_ids: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> RoutingPolicyOutput:
        gnn_output = self.gnn(node_features, edge_index, edge_features)
        return self.act_from_encoding(
            gnn_output,
            current_node=current_node,
            target_node=target_node,
            action_neighbor_ids=action_neighbor_ids,
            action_mask=action_mask,
            deterministic=deterministic,
        )

    def act_from_encoding(
        self,
        gnn_output: GNNEncoderOutput,
        current_node: int,
        target_node: int,
        action_neighbor_ids: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> RoutingPolicyOutput:
        logits = self.action_head(
            gnn_output.node_embeddings,
            gnn_output.graph_embedding,
            gnn_output.batch,
            current_node=int(current_node),
            target_node=int(target_node),
            action_neighbor_ids=action_neighbor_ids,
            action_mask=action_mask,
        )
        distribution = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        value = self.value_head(
            gnn_output.node_embeddings,
            gnn_output.graph_embedding,
            current_node=int(current_node),
            target_node=int(target_node),
        )
        return RoutingPolicyOutput(
            logits=logits,
            action=action,
            log_prob=distribution.log_prob(action),
            entropy=distribution.entropy(),
            value=value,
            gnn_output=gnn_output,
        )
