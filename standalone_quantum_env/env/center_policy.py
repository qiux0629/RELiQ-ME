from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from env.gnn_encoder import CenterSelectionHead, GNNEncoderOutput, QuantumGraphGNNEncoder


@dataclass
class CenterPolicyOutput:
    logits: torch.Tensor
    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    gnn_output: GNNEncoderOutput


class MultipartiteCenterPolicy(nn.Module):
    """GNN policy that selects the GHZ fusion center for a multipartite request."""

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
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
        self.center_head = CenterSelectionHead(hidden_dim, graph_output_dim, dropout=dropout)
        self.value_head = nn.Sequential(
            nn.Linear(graph_output_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> CenterPolicyOutput:
        gnn_output = self.gnn(node_features, edge_index, edge_features)
        logits = self.center_head(
            gnn_output.node_embeddings,
            gnn_output.graph_embedding,
            gnn_output.batch,
            candidate_mask=candidate_mask,
        )
        distribution = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        value = self.value_head(gnn_output.graph_embedding).squeeze(-1)
        return CenterPolicyOutput(
            logits=logits,
            action=action,
            log_prob=distribution.log_prob(action),
            entropy=distribution.entropy(),
            value=value,
            gnn_output=gnn_output,
        )
