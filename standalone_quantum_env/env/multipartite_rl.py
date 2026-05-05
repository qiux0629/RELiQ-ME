from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env.ghz_simulator import GhzSimulatorBackend
from env.multipartite import (
    DelayModel,
    FidelityModel,
    MultipartiteExecution,
    MultipartiteRequest,
    RewardWeights,
    build_gnn_feature_package,
    calculate_center_selection_reward,
    execute_multipartite_request,
    get_valid_center_candidates,
)


@dataclass
class CenterSelectionObservation:
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    candidate_mask: np.ndarray
    candidate_centers: List[int]
    terminals: List[int]
    node_feature_names: List[str]
    edge_feature_names: List[str]

    def to_tensors(self, device: torch.device | str = "cpu") -> Dict[str, torch.Tensor]:
        return {
            "node_features": torch.as_tensor(self.node_features, dtype=torch.float32, device=device),
            "edge_index": torch.as_tensor(self.edge_index, dtype=torch.long, device=device),
            "edge_features": torch.as_tensor(self.edge_features, dtype=torch.float32, device=device),
            "candidate_mask": torch.as_tensor(self.candidate_mask, dtype=torch.bool, device=device),
        }


@dataclass
class PolicySelection:
    center: int
    q_value: float
    candidate_q_values: Dict[int, float]
    observation: CenterSelectionObservation


@dataclass
class Transition:
    observation: CenterSelectionObservation
    action: int
    reward: float


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.storage: Deque[Transition] = deque(maxlen=int(capacity))

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, transition: Transition) -> None:
        self.storage.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(list(self.storage), min(batch_size, len(self.storage)))


def build_center_selection_observation(
    network,
    terminals: Sequence[int],
    center_candidates: Optional[Iterable[int]] = None,
) -> CenterSelectionObservation:
    terminals = [int(node) for node in terminals]
    package = build_gnn_feature_package(network, terminals=terminals, center=None)
    candidate_centers = get_valid_center_candidates(network, terminals, center_candidates)
    candidate_mask = np.zeros(package["node_features"].shape[0], dtype=bool)
    candidate_mask[candidate_centers] = True

    node_features = np.concatenate(
        [
            package["node_features"].astype(np.float32),
            candidate_mask.astype(np.float32)[:, None],
        ],
        axis=1,
    )
    edge_features = package["edge_features"].astype(np.float32)
    if edge_features.ndim == 1:
        edge_features = np.zeros((0, len(package["edge_feature_names"])), dtype=np.float32)

    return CenterSelectionObservation(
        node_features=node_features,
        edge_index=package["edge_index"].astype(np.int64),
        edge_features=edge_features,
        candidate_mask=candidate_mask,
        candidate_centers=[int(center) for center in candidate_centers],
        terminals=terminals,
        node_feature_names=list(package["node_feature_names"]) + ["is_center_candidate"],
        edge_feature_names=list(package["edge_feature_names"]),
    )


class RelationalGnnEncoder(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.node_feature_dim = int(node_feature_dim)
        self.edge_feature_dim = int(edge_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

        self.node_in = nn.Linear(self.node_feature_dim, self.hidden_dim)
        self.edge_in = nn.Linear(self.edge_feature_dim, self.hidden_dim)
        self.message_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim * 2, self.hidden_dim) for _ in range(self.num_layers)]
        )
        self.update_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim * 2, self.hidden_dim) for _ in range(self.num_layers)]
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        node_state = F.relu(self.node_in(node_features))
        if edge_features.numel() == 0:
            edge_state = edge_features.new_zeros((0, self.hidden_dim))
        else:
            edge_state = F.relu(self.edge_in(edge_features))

        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            if edge_index.numel() == 0:
                aggregate = node_state.new_zeros(node_state.shape)
            else:
                source = edge_index[0]
                target = edge_index[1]
                messages = F.relu(message_layer(torch.cat([node_state[source], edge_state], dim=-1)))
                aggregate = node_state.new_zeros(node_state.shape)
                aggregate.index_add_(0, target, messages)
            node_state = F.relu(update_layer(torch.cat([node_state, aggregate], dim=-1)))
        return node_state


class CenterDqnPolicy(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.config = {
            "node_feature_dim": int(node_feature_dim),
            "edge_feature_dim": int(edge_feature_dim),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
        }
        self.encoder = RelationalGnnEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.q_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        node_embeddings = self.encoder(node_features, edge_index, edge_features)
        q_values = self.q_head(node_embeddings).squeeze(-1)
        if candidate_mask is not None:
            q_values = q_values.masked_fill(~candidate_mask.bool(), -1e9)
        return q_values

    @torch.no_grad()
    def act(
        self,
        observation: CenterSelectionObservation,
        epsilon: float = 0.0,
        device: torch.device | str = "cpu",
        rng: Optional[np.random.Generator] = None,
    ) -> PolicySelection:
        if not observation.candidate_centers:
            raise ValueError("No valid center candidates are available.")

        if epsilon > 0:
            rng = rng or np.random.default_rng()
            if rng.random() < epsilon:
                center = int(rng.choice(observation.candidate_centers))
                tensors = observation.to_tensors(device)
                q_values = self(**tensors)
                q_value = float(q_values[center].detach().cpu().item())
                return PolicySelection(
                    center=center,
                    q_value=q_value,
                    candidate_q_values=_candidate_q_values(q_values, observation.candidate_centers),
                    observation=observation,
                )

        tensors = observation.to_tensors(device)
        q_values = self(**tensors)
        center = int(torch.argmax(q_values).detach().cpu().item())
        return PolicySelection(
            center=center,
            q_value=float(q_values[center].detach().cpu().item()),
            candidate_q_values=_candidate_q_values(q_values, observation.candidate_centers),
            observation=observation,
        )


class CenterSelectionEnv:
    def __init__(
        self,
        network,
        request: MultipartiteRequest,
        delay_model: Optional[DelayModel] = None,
        reward_weights: Optional[RewardWeights] = None,
        fidelity_model: Optional[FidelityModel] = None,
        max_hops: Optional[int] = None,
        ghz_simulator: Optional[GhzSimulatorBackend] = None,
    ):
        self.network = network
        self.request = request
        self.delay_model = delay_model or DelayModel()
        self.reward_weights = reward_weights or RewardWeights()
        self.fidelity_model = fidelity_model or FidelityModel()
        self.max_hops = max_hops
        self.ghz_simulator = ghz_simulator

    def reset(self) -> CenterSelectionObservation:
        return build_center_selection_observation(self.network, self.request.terminals)

    def step(self, action: int) -> Tuple[None, float, bool, Dict[str, object]]:
        try:
            execution = execute_multipartite_request(
                self.network,
                self.request,
                center_strategy="rl",
                delay_model=self.delay_model,
                reward_weights=self.reward_weights,
                fidelity_model=self.fidelity_model,
                max_hops=self.max_hops,
                center=int(action),
                ghz_simulator=self.ghz_simulator,
                use_rl_reward=True,
            )
            reward = float(execution.rl_reward if execution.rl_reward is not None else 0.0)
            info: Dict[str, object] = {"execution": execution}
        except ValueError as exc:
            reward = calculate_center_selection_reward(
                path_found=False,
                ghz_success=False,
                ghz_input_fidelity=0.0,
                ghz_fidelity=0.0,
            )
            info = {"error": str(exc)}
        return None, reward, True, info


def _candidate_q_values(q_values: torch.Tensor, candidate_centers: Sequence[int]) -> Dict[int, float]:
    return {
        int(center): float(q_values[int(center)].detach().cpu().item())
        for center in candidate_centers
    }


def select_center_with_policy(
    policy: CenterDqnPolicy,
    network,
    terminals: Sequence[int],
    device: torch.device | str = "cpu",
) -> PolicySelection:
    policy.eval()
    observation = build_center_selection_observation(network, terminals)
    return policy.act(observation, epsilon=0.0, device=device)


def compute_one_step_dqn_loss(
    policy: CenterDqnPolicy,
    transitions: Sequence[Transition],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    losses = []
    for transition in transitions:
        tensors = transition.observation.to_tensors(device)
        q_values = policy(**tensors)
        target = torch.tensor(float(transition.reward), dtype=torch.float32, device=device)
        losses.append(F.mse_loss(q_values[int(transition.action)], target))
    if not losses:
        return torch.tensor(0.0, dtype=torch.float32, device=device)
    return torch.stack(losses).mean()


def save_policy_checkpoint(
    path: Path | str,
    policy: CenterDqnPolicy,
    optimizer: Optional[torch.optim.Optimizer] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    payload = {
        "policy_state_dict": policy.state_dict(),
        "policy_config": policy.config,
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_policy_checkpoint(
    path: Path | str,
    device: torch.device | str = "cpu",
) -> Tuple[CenterDqnPolicy, Dict[str, object]]:
    checkpoint = torch.load(Path(path), map_location=device)
    config = checkpoint["policy_config"]
    policy = CenterDqnPolicy(**config).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    return policy, dict(checkpoint.get("metadata", {}))
