from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from env.quantum_network import QuantumLink


@dataclass(frozen=True)
class DelayModel:
    """Step-equivalent timing model for routing and multipartite entanglement."""

    sample_time: float = 1.0
    packet_time: float = 1.0
    processing_time: float = 1.0
    agent_decision_time: float = 1.0
    gnn_inference_time: float = 1.0
    bsm_time: float = 1.0
    memory_time: float = 1.0
    ghz_fusion_time: float = 1.0
    classical_message_scale: float = 1.0

    def classical_edge_time(self, network, start: int, end: int) -> float:
        distance = network.get_node_distance(network.nodes[start], network.nodes[end])
        speed = (
            network.speed_of_light_glass
            if network.nodes[start].z == 0 and network.nodes[end].z == 0
            else network.speed_of_light_air
        )
        propagation_seconds = distance / speed if speed > 0 else 0.0
        return self.classical_message_scale * propagation_seconds / network.delta_time


@dataclass(frozen=True)
class RewardWeights:
    success_bonus: float = 2.0
    path_found_bonus: float = 0.5
    fidelity_weight: float = 1.0
    latency_weight: float = 0.01
    resource_weight: float = 0.02


@dataclass(frozen=True)
class FidelityModel:
    """RELiQ-style fidelity thresholds plus a GHZ fusion noise approximation."""

    threshold: float = QuantumLink.FIDELITY_THRESHOLD
    ghz_fusion_gate_fidelity: float = 1.0
    ghz_fusion_success_probability: float = 1.0


@dataclass(frozen=True)
class StateCarvingGHZModel:
    """Local GHZ generation model used by the RELiQ multipartite extension."""

    profile: str = "state_carving_2025_high_fidelity"
    success_probability: float = 1.0
    fidelity: float = 0.9999
    local_gate_fidelity: float = 0.9999
    generation_time: float = 1.0
    construction: str = "sequential"

    @classmethod
    def from_profile(cls, profile: str = "state_carving_2025_high_fidelity", **overrides) -> "StateCarvingGHZModel":
        if profile not in {"state_carving_2025_high_fidelity", "ideal"}:
            raise ValueError(f"Unknown state-carving GHZ profile: {profile}")
        values = {
            "profile": profile,
            "success_probability": 1.0,
            "fidelity": 0.9999,
            "local_gate_fidelity": 0.9999,
            "generation_time": 1.0,
            "construction": "sequential",
        }
        if profile == "ideal":
            values["fidelity"] = 1.0
            values["local_gate_fidelity"] = 1.0
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)

    def ghz_success_probability(self, n_qubits: int) -> float:
        n_qubits = max(int(n_qubits), 2)
        base = max(0.0, min(1.0, float(self.success_probability)))
        if self.construction == "direct":
            return base
        return float(base ** max(n_qubits - 1, 1))

    def ghz_fidelity(self, n_qubits: int) -> float:
        n_qubits = max(int(n_qubits), 2)
        base_fidelity = max(0.0, min(1.0, float(self.fidelity)))
        gate_fidelity = max(0.0, min(1.0, float(self.local_gate_fidelity)))
        if self.construction == "direct":
            return base_fidelity
        carving_steps = max(n_qubits - 1, 1)
        local_gates = max(n_qubits - 2, 0)
        return float((base_fidelity ** carving_steps) * (gate_fidelity ** local_gates))


@dataclass(frozen=True)
class PhysicalMemoryModel:
    """Photon-atom memory interface parameters for dynamic physical execution."""

    profile: str = "single_atom_cavity_2015"
    photon_to_atom_success_probability: float = 0.39
    photon_to_atom_fidelity: float = 0.86
    atom_to_photon_success_probability: float = 0.69
    atom_to_photon_fidelity: float = 0.88
    memory_lifetime_seconds: float = 0.22
    max_write_attempts: int = 1
    max_readout_attempts: int = 1
    memory_slots_per_node: int = 0
    max_photonic_route_attempts: int = 1
    purification_rounds: int = 0
    purification_success_probability: float = 0.9
    purification_fidelity_gain: float = 0.02

    @classmethod
    def from_profile(cls, profile: str = "single_atom_cavity_2015", **overrides) -> "PhysicalMemoryModel":
        if profile not in {"single_atom_cavity_2015", "state_carving_2025_high_fidelity"}:
            raise ValueError(f"Unknown physical memory profile: {profile}")
        values = {
            "profile": profile,
            "photon_to_atom_success_probability": 0.39,
            "photon_to_atom_fidelity": 0.86,
            "atom_to_photon_success_probability": 0.69,
            "atom_to_photon_fidelity": 0.88,
            "memory_lifetime_seconds": 0.22,
            "max_write_attempts": 1,
            "max_readout_attempts": 1,
            "memory_slots_per_node": 0,
            "max_photonic_route_attempts": 1,
            "purification_rounds": 0,
            "purification_success_probability": 0.9,
            "purification_fidelity_gain": 0.02,
        }
        if profile == "state_carving_2025_high_fidelity":
            values["photon_to_atom_fidelity"] = 0.9999
        values.update({key: value for key, value in overrides.items() if value is not None})
        for key in (
            "max_write_attempts",
            "max_readout_attempts",
            "memory_slots_per_node",
            "max_photonic_route_attempts",
            "purification_rounds",
        ):
            values[key] = int(values[key])
        return cls(**values)

    @staticmethod
    def apply_interface_fidelity(input_fidelity: float, interface_fidelity: float) -> float:
        shrink = max(0.0, min(1.0, 2.0 * interface_fidelity - 1.0))
        return float(0.5 + (input_fidelity - 0.5) * shrink)

    def apply_memory_decay(self, input_fidelity: float, wait_seconds: float) -> float:
        if self.memory_lifetime_seconds <= 0:
            return 0.5
        return float(0.5 + (input_fidelity - 0.5) * np.exp(-max(wait_seconds, 0.0) / self.memory_lifetime_seconds))

    def apply_purification(self, input_fidelity: float, rng) -> Tuple[float, bool, int]:
        fidelity = float(input_fidelity)
        completed_rounds = 0
        for _ in range(max(int(self.purification_rounds), 0)):
            if rng.random() >= self.purification_success_probability:
                return fidelity, False, completed_rounds
            fidelity = min(1.0, fidelity + self.purification_fidelity_gain * (1.0 - fidelity))
            completed_rounds += 1
        return float(fidelity), True, completed_rounds


@dataclass
class StoredAtomicEntanglement:
    terminal: int
    center: int
    stored_at_time: float
    initial_fidelity: float
    write_success: bool
    write_attempts: int = 0
    memory_wait_time: float = 0.0
    decayed_fidelity: float = 0.0
    readout_success: bool = False
    readout_attempts: int = 0
    readout_fidelity: float = 0.0
    purification_success: Optional[bool] = None
    purification_rounds_completed: int = 0
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "terminal": int(self.terminal),
            "center": int(self.center),
            "stored_at_time": float(self.stored_at_time),
            "initial_fidelity": float(self.initial_fidelity),
            "write_success": bool(self.write_success),
            "write_attempts": int(self.write_attempts),
            "memory_wait_time": float(self.memory_wait_time),
            "decayed_fidelity": float(self.decayed_fidelity),
            "readout_success": bool(self.readout_success),
            "readout_attempts": int(self.readout_attempts),
            "readout_fidelity": float(self.readout_fidelity),
            "purification_success": self.purification_success,
            "purification_rounds_completed": int(self.purification_rounds_completed),
            "failure_reason": self.failure_reason,
        }


@dataclass
class PathEstimate:
    terminal: int
    center: int
    path: List[int]
    hop_count: int
    quantum_generation_time: float
    classical_signal_time: float
    agent_time: float
    coordination_time: float
    swap_time: float
    total_time: float
    fidelity: float
    available_links: int
    physical_metadata: Optional[Dict[str, object]] = None

    def to_dict(self) -> Dict[str, object]:
        result = asdict(self)
        for key, value in list(result.items()):
            if isinstance(value, np.generic):
                result[key] = value.item()
        return result


@dataclass
class MultipartitePlan:
    terminals: List[int]
    center: int
    center_strategy: str
    paths: List[PathEstimate]
    total_time: float
    total_hops: int
    bottleneck_fidelity: float
    ghz_input_fidelity: float
    ghz_fidelity: float
    fusion_gate_fidelity: float
    fusion_success_probability: float
    success_probability: float
    reward: float
    ghz_establishment_mode: str = "fusion_after_bipartite"
    local_ghz_fidelity: float = 1.0
    local_ghz_success_probability: float = 1.0
    local_ghz_generation_time: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "terminals": self.terminals,
            "center": self.center,
            "center_strategy": self.center_strategy,
            "ghz_establishment_mode": self.ghz_establishment_mode,
            "total_time": self.total_time,
            "total_hops": self.total_hops,
            "bottleneck_fidelity": self.bottleneck_fidelity,
            "ghz_input_fidelity": self.ghz_input_fidelity,
            "ghz_fidelity": self.ghz_fidelity,
            "fusion_gate_fidelity": self.fusion_gate_fidelity,
            "fusion_success_probability": self.fusion_success_probability,
            "local_ghz_fidelity": self.local_ghz_fidelity,
            "local_ghz_success_probability": self.local_ghz_success_probability,
            "local_ghz_generation_time": self.local_ghz_generation_time,
            "success_probability": self.success_probability,
            "reward": self.reward,
            "paths": [path.to_dict() for path in self.paths],
        }


@dataclass
class MultipartiteRequest:
    id: int
    source: int
    targets: List[int]
    ttl: Optional[int] = None

    @property
    def terminals(self) -> List[int]:
        return [self.source] + self.targets

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": int(self.id),
            "source": int(self.source),
            "targets": [int(target) for target in self.targets],
            "terminals": [int(terminal) for terminal in self.terminals],
            "ttl": None if self.ttl is None else int(self.ttl),
        }


@dataclass
class RouteExecution:
    role: str
    terminal: int
    target_index: Optional[int]
    path: List[int]
    hop_count: int
    total_time: float
    fidelity: float
    completed: bool
    failure_reason: Optional[str]
    physical_metadata: Optional[Dict[str, object]] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "role": self.role,
            "terminal": int(self.terminal),
            "target_index": self.target_index,
            "path": [int(node) for node in self.path],
            "hop_count": int(self.hop_count),
            "total_time": float(self.total_time),
            "fidelity": float(self.fidelity),
            "completed": bool(self.completed),
            "failure_reason": self.failure_reason,
            "physical_metadata": self.physical_metadata,
        }


@dataclass
class MultipartiteExecution:
    request: MultipartiteRequest
    plan: MultipartitePlan
    routes: List[RouteExecution]
    center_wait_time: float
    ghz_fusion_time: float
    total_time: float
    ghz_input_fidelity: float
    ghz_fidelity: float
    success_probability: float
    fusion_sampled_probability: float
    fusion_succeeded: bool
    success: bool
    failure_reason: Optional[str]
    bipartite_backend: str = "path_estimator"
    ghz_establishment_mode: str = "fusion_after_bipartite"
    local_ghz_sampled_probability: float = 1.0
    local_ghz_succeeded: bool = True
    physical_timeline: Optional[List[Dict[str, object]]] = None
    stored_entanglements: Optional[List[StoredAtomicEntanglement]] = None
    memory_wait_time: float = 0.0
    memory_decay_loss: float = 0.0
    write_success: Optional[bool] = None
    readout_success: Optional[bool] = None
    conversion_failure_rate: float = 0.0
    memory_failure_rate: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        if self.ghz_establishment_mode == "prebuilt_ghz_distribution":
            flow = [
                "one_to_many_request_generation",
                "reliq_style_center_selection",
                "local_state_carving_ghz_generation",
                "ghz_particle_distribution_to_terminals",
                "multipartite_request_success_or_failure",
            ]
        else:
            flow = [
                "one_to_many_request_generation",
                "reliq_style_center_selection",
                "terminal_to_center_bipartite_routing",
                "center_waits_for_all_bipartite_links",
                "ghz_fusion",
                "multipartite_request_success_or_failure",
            ]
        return {
            "flow": flow,
            "request": self.request.to_dict(),
            "center": int(self.plan.center),
            "center_strategy": self.plan.center_strategy,
            "ghz_establishment_mode": self.ghz_establishment_mode,
            "center_wait_time": float(self.center_wait_time),
            "ghz_fusion_time": float(self.ghz_fusion_time),
            "local_ghz_generation_time": float(self.plan.local_ghz_generation_time),
            "local_ghz_fidelity": float(self.plan.local_ghz_fidelity),
            "local_ghz_success_probability": float(self.plan.local_ghz_success_probability),
            "local_ghz_sampled_probability": float(self.local_ghz_sampled_probability),
            "local_ghz_succeeded": bool(self.local_ghz_succeeded),
            "total_time": float(self.total_time),
            "ghz_input_fidelity": float(self.ghz_input_fidelity),
            "ghz_fidelity": float(self.ghz_fidelity),
            "success_probability": float(self.success_probability),
            "fusion_sampled_probability": float(self.fusion_sampled_probability),
            "fusion_succeeded": bool(self.fusion_succeeded),
            "success": bool(self.success),
            "failure_reason": self.failure_reason,
            "bipartite_backend": self.bipartite_backend,
            "physical_timeline": self.physical_timeline or [],
            "stored_entanglements": [
                entanglement.to_dict() for entanglement in (self.stored_entanglements or [])
            ],
            "memory_wait_time": float(self.memory_wait_time),
            "memory_decay_loss": float(self.memory_decay_loss),
            "write_success": self.write_success,
            "readout_success": self.readout_success,
            "conversion_failure_rate": float(self.conversion_failure_rate),
            "memory_failure_rate": float(self.memory_failure_rate),
            "routes": [route.to_dict() for route in self.routes],
            "plan": self.plan.to_dict(),
        }


def _edge_for_nodes(network, start: int, end: int):
    return network.edge_node_association[(min(start, end), max(start, end))]


def _edge_fidelities(edge) -> List[float]:
    return [float(link.fidelity) for link in edge.links if link.fidelity > 0]


def _edge_best_fidelity(edge, fallback: float) -> float:
    fidelities = _edge_fidelities(edge)
    if not fidelities:
        return fallback
    return max(fidelities)


def _edge_available_links(edge) -> int:
    return max(len(edge.links) - edge.get_total_reservations(), 0)


def _edge_generation_time(network, edge) -> float:
    if _edge_available_links(edge) > 0:
        return 0.0
    probabilities = network.get_link_entanglement_probability(
        edge.start,
        edge.end,
        edge.intermediate_routers,
        free_slots=1,
    )
    success_probability = max(1.0 - probabilities[0], 1e-9)
    return 1.0 / success_probability


def _path_has_quantum_resources(network, path: Sequence[int]) -> bool:
    for start, end in zip(path, path[1:]):
        edge = _edge_for_nodes(network, start, end)
        if _edge_available_links(edge) <= 0:
            return False
    return True


def _apply_depolarizing_gate(
    input_fidelity: float,
    gate_fidelity: float,
    n_qubits: int,
) -> float:
    dimension = 2 ** max(n_qubits, 1)
    if gate_fidelity >= 1.0:
        return float(input_fidelity)
    if gate_fidelity <= 1.0 / dimension:
        return float(1.0 / dimension)

    depolarization_lambda = dimension / (dimension - 1) * (1.0 - gate_fidelity)
    depolarization_lambda = min(max(depolarization_lambda, 0.0), 1.0)
    return float((1.0 - depolarization_lambda) * input_fidelity + depolarization_lambda / dimension)


def estimate_ghz_fidelity(
    path_fidelities: Sequence[float],
    fidelity_model: FidelityModel,
) -> Tuple[float, float]:
    if not path_fidelities:
        return 0.0, 0.0

    ghz_input_fidelity = float(np.prod(path_fidelities))
    ghz_fidelity = _apply_depolarizing_gate(
        ghz_input_fidelity,
        fidelity_model.ghz_fusion_gate_fidelity,
        n_qubits=len(path_fidelities),
    )
    if ghz_fidelity < fidelity_model.threshold:
        ghz_fidelity = 0.0
    return ghz_input_fidelity, ghz_fidelity


def estimate_path_fidelity(network, path: Sequence[int]) -> float:
    if len(path) < 2:
        return 1.0

    current_fidelity: Optional[float] = None
    for index in range(len(path) - 1):
        edge = _edge_for_nodes(network, path[index], path[index + 1])
        edge_fidelity = _edge_best_fidelity(edge, network.initial_fidelity)
        if current_fidelity is None:
            current_fidelity = edge_fidelity
            continue

        swap_node = path[index]
        gate_error = 1.0 - network.nodes[swap_node].swap_prob
        current_fidelity = QuantumLink.get_swap_fidelity(
            current_fidelity,
            edge_fidelity,
            gate_error=gate_error,
        )
        if current_fidelity <= 0:
            return 0.0

    return float(current_fidelity if current_fidelity is not None else 0.0)


def estimate_path_from_nodes(
    network,
    terminal: int,
    center: int,
    path: Sequence[int],
    delay_model: DelayModel,
) -> PathEstimate:
    quantum_generation_time = 0.0
    classical_signal_time = 0.0
    available_links = 0

    for start, end in zip(path, path[1:]):
        edge = _edge_for_nodes(network, start, end)
        quantum_generation_time += _edge_generation_time(network, edge)
        classical_signal_time += delay_model.classical_edge_time(network, start, end)
        available_links += _edge_available_links(edge)

    hop_count = max(len(path) - 1, 0)
    agent_time = delay_model.gnn_inference_time + hop_count * delay_model.agent_decision_time
    coordination_time = (
        delay_model.sample_time
        + delay_model.packet_time
        + delay_model.processing_time
        + classical_signal_time
    )
    swap_time = max(hop_count - 1, 0) * (
        delay_model.bsm_time + delay_model.memory_time + classical_signal_time / max(hop_count, 1)
    )
    total_time = quantum_generation_time + coordination_time + agent_time + swap_time

    return PathEstimate(
        terminal=int(terminal),
        center=int(center),
        path=[int(node) for node in path],
        hop_count=int(hop_count),
        quantum_generation_time=float(quantum_generation_time),
        classical_signal_time=float(classical_signal_time),
        agent_time=float(agent_time),
        coordination_time=float(coordination_time),
        swap_time=float(swap_time),
        total_time=float(total_time),
        fidelity=estimate_path_fidelity(network, path),
        available_links=int(available_links),
    )


def estimate_path(
    network,
    terminal: int,
    center: int,
    delay_model: DelayModel,
) -> PathEstimate:
    path = nx.shortest_path(network.G, terminal, center, weight=network.get_edge_weight_binary)
    return estimate_path_from_nodes(network, terminal, center, path, delay_model)


def _as_distribution_path(network, path_estimate: PathEstimate, delay_model: DelayModel) -> PathEstimate:
    """Represent the same route in center-to-terminal direction for prebuilt GHZ distribution."""

    if path_estimate.path and path_estimate.path[0] == path_estimate.center:
        return path_estimate
    reversed_path = list(reversed(path_estimate.path))
    distribution = estimate_path_from_nodes(
        network,
        path_estimate.terminal,
        path_estimate.center,
        reversed_path,
        delay_model,
    )
    distribution.physical_metadata = path_estimate.physical_metadata
    return distribution


def _as_distribution_paths(network, paths: Sequence[PathEstimate], delay_model: DelayModel) -> List[PathEstimate]:
    return [_as_distribution_path(network, path, delay_model) for path in paths]


def _build_plan_from_paths(
    terminals: Sequence[int],
    center: int,
    center_strategy: str,
    paths: Sequence[PathEstimate],
    delay_model: DelayModel,
    reward_weights: RewardWeights,
    fidelity_model: FidelityModel,
    ghz_establishment_mode: str = "fusion_after_bipartite",
    state_carving_model: Optional[StateCarvingGHZModel] = None,
) -> MultipartitePlan:
    state_carving_model = state_carving_model or StateCarvingGHZModel.from_profile()
    route_time = max((path.total_time for path in paths), default=0.0)
    total_hops = sum(path.hop_count for path in paths)
    path_fidelities = [path.fidelity for path in paths]
    bottleneck_fidelity = min(path_fidelities, default=0.0)

    if ghz_establishment_mode == "prebuilt_ghz_distribution":
        local_ghz_fidelity = state_carving_model.ghz_fidelity(len(terminals))
        local_ghz_success_probability = state_carving_model.ghz_success_probability(len(terminals))
        local_ghz_generation_time = float(state_carving_model.generation_time)
        distribution_fidelity = float(np.prod(path_fidelities)) if path_fidelities else 0.0
        ghz_input_fidelity = float(local_ghz_fidelity)
        ghz_fidelity = float(local_ghz_fidelity * distribution_fidelity)
        if ghz_fidelity < fidelity_model.threshold:
            ghz_fidelity = 0.0
        success_probability = (
            local_ghz_success_probability
            if bottleneck_fidelity >= fidelity_model.threshold and ghz_fidelity >= fidelity_model.threshold
            else 0.0
        )
        total_time = route_time + local_ghz_generation_time
        fusion_gate_fidelity = 1.0
        fusion_success_probability = 0.0
    elif ghz_establishment_mode == "fusion_after_bipartite":
        local_ghz_fidelity = 1.0
        local_ghz_success_probability = 1.0
        local_ghz_generation_time = 0.0
        ghz_input_fidelity, ghz_fidelity = estimate_ghz_fidelity(path_fidelities, fidelity_model)
        success_probability = (
            fidelity_model.ghz_fusion_success_probability
            if bottleneck_fidelity >= fidelity_model.threshold and ghz_fidelity >= fidelity_model.threshold
            else 0.0
        )
        total_time = route_time + delay_model.ghz_fusion_time
        fusion_gate_fidelity = fidelity_model.ghz_fusion_gate_fidelity
        fusion_success_probability = fidelity_model.ghz_fusion_success_probability
    else:
        raise ValueError(f"Unknown GHZ establishment mode: {ghz_establishment_mode}")

    reward = (
        reward_weights.success_bonus * success_probability
        + reward_weights.path_found_bonus
        + reward_weights.fidelity_weight * ghz_fidelity
        - reward_weights.latency_weight * total_time
        - reward_weights.resource_weight * total_hops
    )
    return MultipartitePlan(
        terminals=[int(terminal) for terminal in terminals],
        center=int(center),
        center_strategy=center_strategy,
        paths=list(paths),
        total_time=float(total_time),
        total_hops=int(total_hops),
        bottleneck_fidelity=float(bottleneck_fidelity),
        ghz_input_fidelity=float(ghz_input_fidelity),
        ghz_fidelity=float(ghz_fidelity),
        fusion_gate_fidelity=float(fusion_gate_fidelity),
        fusion_success_probability=float(fusion_success_probability),
        success_probability=float(success_probability),
        reward=float(reward),
        ghz_establishment_mode=ghz_establishment_mode,
        local_ghz_fidelity=float(local_ghz_fidelity),
        local_ghz_success_probability=float(local_ghz_success_probability),
        local_ghz_generation_time=float(local_ghz_generation_time),
    )


def _valid_centers(network, terminals: Sequence[int], candidates: Optional[Iterable[int]]) -> List[int]:
    if candidates is None:
        candidates = range(network.n_nodes_connected)

    terminal_set = set(terminals)
    centers = []
    for node in candidates:
        if node in terminal_set:
            continue
        if node >= len(network.nodes) or network.nodes[node].virtual:
            continue
        if not nx.has_path(network.G, node, terminals[0]):
            continue
        if all(nx.has_path(network.G, node, terminal) for terminal in terminals):
            centers.append(int(node))
    return centers


def valid_center_candidates(
    network,
    terminals: Sequence[int],
    candidates: Optional[Iterable[int]] = None,
) -> List[int]:
    return _valid_centers(network, terminals, candidates)


def _score_plan(
    paths: Sequence[PathEstimate],
    center: int,
    strategy: str,
    delay_model: DelayModel,
    reward_weights: RewardWeights,
    fidelity_model: FidelityModel,
    ghz_establishment_mode: str = "fusion_after_bipartite",
    state_carving_model: Optional[StateCarvingGHZModel] = None,
) -> Tuple[float, float, float, float, float, int]:
    pseudo_plan = _build_plan_from_paths(
        terminals=[path.terminal for path in paths],
        center=center,
        center_strategy=strategy,
        paths=paths,
        delay_model=delay_model,
        reward_weights=reward_weights,
        fidelity_model=fidelity_model,
        ghz_establishment_mode=ghz_establishment_mode,
        state_carving_model=state_carving_model,
    )

    if strategy == "median":
        score = -pseudo_plan.total_hops
    elif strategy == "minimax":
        score = -max((path.hop_count for path in paths), default=0)
    elif strategy == "max-fidelity":
        score = pseudo_plan.ghz_fidelity
    elif strategy == "balanced":
        score = pseudo_plan.reward
    elif strategy == "min-latency":
        score = -pseudo_plan.total_time
    else:
        score = pseudo_plan.reward

    return (
        score,
        pseudo_plan.total_time,
        pseudo_plan.bottleneck_fidelity,
        pseudo_plan.ghz_input_fidelity,
        pseudo_plan.ghz_fidelity,
        pseudo_plan.total_hops,
    )


def plan_multipartite_entanglement(
    network,
    terminals: Sequence[int],
    center_strategy: str = "balanced",
    delay_model: Optional[DelayModel] = None,
    reward_weights: Optional[RewardWeights] = None,
    fidelity_model: Optional[FidelityModel] = None,
    ghz_establishment_mode: str = "prebuilt_ghz_distribution",
    state_carving_model: Optional[StateCarvingGHZModel] = None,
    center_candidates: Optional[Iterable[int]] = None,
) -> MultipartitePlan:
    if len(terminals) < 2:
        raise ValueError("At least two terminals are required.")

    delay_model = delay_model or DelayModel()
    reward_weights = reward_weights or RewardWeights()
    fidelity_model = fidelity_model or FidelityModel()
    state_carving_model = state_carving_model or StateCarvingGHZModel.from_profile()
    terminals = [int(node) for node in terminals]
    centers = _valid_centers(network, terminals, center_candidates)
    if not centers:
        raise ValueError("No reachable center candidate found for terminals.")

    if center_strategy == "random":
        center = int(centers[network.topology_generator.integers(len(centers))])
        paths = [estimate_path(network, terminal, center, delay_model) for terminal in terminals]
        if ghz_establishment_mode == "prebuilt_ghz_distribution":
            paths = _as_distribution_paths(network, paths, delay_model)
    else:
        best = None
        for candidate in centers:
            paths = [estimate_path(network, terminal, candidate, delay_model) for terminal in terminals]
            if ghz_establishment_mode == "prebuilt_ghz_distribution":
                paths = _as_distribution_paths(network, paths, delay_model)
            score, total_time, bottleneck_fidelity, ghz_input_fidelity, ghz_fidelity, total_hops = _score_plan(
                paths,
                candidate,
                center_strategy,
                delay_model,
                reward_weights,
                fidelity_model,
                ghz_establishment_mode=ghz_establishment_mode,
                state_carving_model=state_carving_model,
            )
            tie_breaker = (-total_time, ghz_fidelity, -total_hops, -candidate)
            current = (score, tie_breaker, candidate, paths, total_time, bottleneck_fidelity, ghz_input_fidelity, ghz_fidelity, total_hops)
            if best is None or current[:2] > best[:2]:
                best = current
        _, _, center, paths, _, _, _, _, _ = best

    return _build_plan_from_paths(
        terminals,
        int(center),
        center_strategy,
        paths,
        delay_model,
        reward_weights,
        fidelity_model,
        ghz_establishment_mode=ghz_establishment_mode,
        state_carving_model=state_carving_model,
    )


def estimate_path_with_routing_policy(
    network,
    terminal: int,
    center: int,
    delay_model: DelayModel,
    routing_policy,
    max_hops: Optional[int] = None,
    device=None,
    deterministic: bool = True,
    cache_static_encoding: bool = False,
    prefer_progress_actions: bool = False,
) -> PathEstimate:
    import torch

    from env.gnn_encoder import GraphBatch, adapt_feature_package_dimensions
    from env.routing_policy import (
        action_neighbor_nodes,
        build_bipartite_feature_package,
        valid_routing_action_mask,
    )

    if routing_policy is None:
        raise ValueError("routing_policy is required when bipartite_backend='routing_policy'.")

    current = int(terminal)
    center = int(center)
    path = [current]
    visited = {current}
    max_steps = int(max_hops or max(network.n_nodes_connected, 1))
    routing_policy.eval()
    cached_gnn_output = None
    with torch.no_grad():
        if cache_static_encoding:
            features = build_bipartite_feature_package(network, current=current, target=center, visited=path)
            expected_node_dim = routing_policy.gnn.node_projection[0].in_features
            expected_edge_dim = routing_policy.gnn.layers[0].edge_encoder[0].in_features
            features = adapt_feature_package_dimensions(features, expected_node_dim, expected_edge_dim)
            batch = GraphBatch.from_feature_package(features, device=device)
            cached_gnn_output = routing_policy.gnn(
                batch.node_features,
                batch.edge_index,
                batch.edge_features,
            )
        for _ in range(max_steps):
            if current == center:
                break
            action_mask = valid_routing_action_mask(
                network,
                current,
                center,
                path,
                device=device,
                prefer_progress=prefer_progress_actions,
            )
            neighbor_ids = action_neighbor_nodes(network, current, device=device)
            if cached_gnn_output is None:
                features = build_bipartite_feature_package(network, current=current, target=center, visited=path)
                expected_node_dim = routing_policy.gnn.node_projection[0].in_features
                expected_edge_dim = routing_policy.gnn.layers[0].edge_encoder[0].in_features
                features = adapt_feature_package_dimensions(features, expected_node_dim, expected_edge_dim)
                batch = GraphBatch.from_feature_package(features, device=device)
                output = routing_policy(
                    batch.node_features,
                    batch.edge_index,
                    batch.edge_features,
                    current_node=current,
                    target_node=center,
                    action_neighbor_ids=neighbor_ids,
                    action_mask=action_mask,
                    deterministic=deterministic,
                )
            else:
                output = routing_policy.act_from_encoding(
                    cached_gnn_output,
                    current_node=current,
                    target_node=center,
                    action_neighbor_ids=neighbor_ids,
                    action_mask=action_mask,
                    deterministic=deterministic,
                )
            action = int(output.action.detach().cpu())
            if action <= 0:
                break
            edge_slot = action - 1
            current_node = network.get_nodes(0)[current]
            if edge_slot >= len(current_node.edges):
                break
            edge = network.edges[current_node.edges[edge_slot]]
            next_node = int(edge.get_other_node(current))
            if next_node in visited:
                break
            path.append(next_node)
            visited.add(next_node)
            current = next_node

    return estimate_path_from_nodes(network, terminal, center, path, delay_model)


def _select_dynamic_route_path(
    network,
    terminal: int,
    center: int,
    delay_model: DelayModel,
    routing_policy=None,
    max_hops: Optional[int] = None,
    device=None,
    deterministic: bool = True,
    cache_static_encoding: bool = False,
    prefer_progress_actions: bool = False,
) -> List[int]:
    if routing_policy is None:
        return [
            int(node)
            for node in nx.shortest_path(
                network.G,
                int(terminal),
                int(center),
                weight=network.get_edge_weight_binary,
            )
        ]
    return estimate_path_with_routing_policy(
        network,
        terminal,
        center,
        delay_model,
        routing_policy,
        max_hops=max_hops,
        device=device,
        deterministic=deterministic,
        cache_static_encoding=cache_static_encoding,
        prefer_progress_actions=prefer_progress_actions,
    ).path


def _dynamic_route_path_candidates(
    network,
    terminal: int,
    center: int,
    delay_model: DelayModel,
    candidate_count: int,
    routing_policy=None,
    max_hops: Optional[int] = None,
    device=None,
    deterministic: bool = True,
    cache_static_encoding: bool = False,
    prefer_progress_actions: bool = False,
) -> List[List[int]]:
    candidate_count = max(int(candidate_count), 1)
    terminal = int(terminal)
    center = int(center)
    raw_paths: List[List[int]] = []

    if routing_policy is not None:
        raw_paths.append(
            _select_dynamic_route_path(
                network,
                terminal,
                center,
                delay_model,
                routing_policy=routing_policy,
                max_hops=max_hops,
                device=device,
                deterministic=deterministic,
                cache_static_encoding=cache_static_encoding,
                prefer_progress_actions=prefer_progress_actions,
            )
        )

    try:
        raw_paths.append(
            [
                int(node)
                for node in nx.shortest_path(
                    network.G,
                    terminal,
                    center,
                    weight=network.get_edge_weight_binary,
                )
            ]
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass

    try:
        for path in nx.shortest_simple_paths(
            network.G,
            terminal,
            center,
            weight=network.get_edge_weight_binary,
        ):
            raw_paths.append([int(node) for node in path])
            if len(raw_paths) >= candidate_count * 3:
                break
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass

    seen: set[Tuple[int, ...]] = set()
    valid_paths: List[List[int]] = []
    for path in raw_paths:
        if len(path) < 2 or path[0] != terminal or path[-1] != center:
            continue
        if max_hops is not None and len(path) - 1 > max_hops:
            continue
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        valid_paths.append(path)

    valid_paths.sort(
        key=lambda path: (
            estimate_path_fidelity(network, path),
            -len(path),
        ),
        reverse=True,
    )
    return valid_paths[:candidate_count]


def execute_dynamic_physical_path(
    network,
    terminal: int,
    center: int,
    delay_model: DelayModel,
    physical_memory_model: PhysicalMemoryModel,
    routing_policy=None,
    max_hops: Optional[int] = None,
    device=None,
    deterministic: bool = True,
    routing_cache_static_encoding: bool = False,
    routing_prefer_progress_actions: bool = False,
) -> PathEstimate:
    candidate_paths = _dynamic_route_path_candidates(
        network,
        terminal,
        center,
        delay_model,
        candidate_count=max(int(physical_memory_model.max_photonic_route_attempts), 1),
        routing_policy=routing_policy,
        max_hops=max_hops,
        device=device,
        deterministic=deterministic,
        cache_static_encoding=routing_cache_static_encoding,
        prefer_progress_actions=routing_prefer_progress_actions,
    )
    max_steps = int(max_hops or max(network.ttl, network.n_nodes_connected, 1))
    elapsed_steps = 0
    timeline: List[Dict[str, object]] = []
    consumed_links = 0
    current_link: Optional[QuantumLink] = None
    best_candidate_link: Optional[QuantumLink] = None
    best_candidate_path: List[int] = candidate_paths[0] if candidate_paths else [int(terminal)]
    best_candidate_fidelity = 0.0
    failure_reason = None
    route_attempts = 0

    if not candidate_paths:
        failure_reason = "dynamic_route_did_not_reach_center"

    if failure_reason is None:
        for attempt in range(max(int(physical_memory_model.max_photonic_route_attempts), 1)):
            path = candidate_paths[min(attempt, len(candidate_paths) - 1)]
            route_attempts = attempt + 1
            attempt_elapsed_steps = 0
            attempt_link: Optional[QuantumLink] = None
            attempt_failure = None
            timeline.append(
                {
                    "event": "photonic_route_attempt_start",
                    "terminal": int(terminal),
                    "center": int(center),
                    "attempt": int(route_attempts),
                    "path": [int(node) for node in path],
                    "time": float(elapsed_steps),
                }
            )
            for hop_index, (start, end) in enumerate(zip(path, path[1:])):
                edge = _edge_for_nodes(network, start, end)
                waited_steps = 0
                while _edge_available_links(edge) <= 0 and attempt_elapsed_steps < max_steps:
                    network.pre_step()
                    network.step()
                    waited_steps += 1
                    elapsed_steps += 1
                    attempt_elapsed_steps += 1
                if _edge_available_links(edge) <= 0:
                    attempt_failure = "dynamic_link_generation_timeout"
                    timeline.append(
                        {
                            "event": "link_generation_timeout",
                            "edge": [int(start), int(end)],
                            "attempt": int(route_attempts),
                            "waited_steps": waited_steps,
                            "time": float(elapsed_steps),
                        }
                    )
                    break

                selected_link = edge.links[-1]
                edge.links.remove(selected_link)
                network.total_active_quantum_links = max(network.total_active_quantum_links - 1, 0)
                consumed_links += 1
                elapsed_steps += 1
                attempt_elapsed_steps += 1
                timeline.append(
                    {
                        "event": "consume_elementary_link",
                        "edge": [int(start), int(end)],
                        "attempt": int(route_attempts),
                        "hop_index": hop_index,
                        "waited_steps": waited_steps,
                        "link_fidelity": float(selected_link.fidelity),
                        "time": float(elapsed_steps),
                    }
                )

                if attempt_link is None:
                    attempt_link = selected_link
                    continue

                swap_node = int(start)
                attempt_link = QuantumLink.swap(
                    attempt_link,
                    selected_link,
                    network.nodes[swap_node].swap_prob,
                    source=swap_node,
                    rng_generator=network.swap_generator,
                )
                timeline.append(
                    {
                        "event": "swap",
                        "node": swap_node,
                        "attempt": int(route_attempts),
                        "fidelity": 0.0 if attempt_link is None else float(attempt_link.fidelity),
                        "time": float(elapsed_steps),
                    }
                )
                if attempt_link is None or attempt_link.fidelity <= 0:
                    attempt_failure = "dynamic_swap_failed"
                    break

            if attempt_failure is None and attempt_link is not None:
                if attempt_link.fidelity >= QuantumLink.FIDELITY_THRESHOLD:
                    if attempt_link.fidelity > best_candidate_fidelity:
                        best_candidate_link = attempt_link
                        best_candidate_path = [int(node) for node in path]
                        best_candidate_fidelity = float(attempt_link.fidelity)
                    failure_reason = None
                    timeline.append(
                        {
                            "event": "photonic_route_attempt_succeeded",
                            "terminal": int(terminal),
                            "center": int(center),
                            "attempt": int(route_attempts),
                            "photonic_fidelity": float(attempt_link.fidelity),
                            "best_photonic_fidelity": float(best_candidate_fidelity),
                            "time": float(elapsed_steps),
                        }
                    )
                    continue
                attempt_failure = "dynamic_photonic_fidelity_below_threshold"

            failure_reason = attempt_failure or "dynamic_route_failed"
            timeline.append(
                {
                    "event": "photonic_route_attempt_failed",
                    "terminal": int(terminal),
                    "center": int(center),
                    "attempt": int(route_attempts),
                    "failure_reason": failure_reason,
                    "time": float(elapsed_steps),
                }
            )
        if best_candidate_link is not None:
            current_link = best_candidate_link
            failure_reason = None
            timeline.append(
                {
                    "event": "photonic_route_candidate_selected",
                    "terminal": int(terminal),
                    "center": int(center),
                    "attempts": int(route_attempts),
                    "path": [int(node) for node in best_candidate_path],
                    "photonic_fidelity": float(best_candidate_fidelity),
                    "time": float(elapsed_steps),
                }
            )
        elif current_link is None and failure_reason is None:
            failure_reason = "dynamic_route_failed"

    photonic_fidelity = float(current_link.fidelity) if current_link is not None and failure_reason is None else 0.0
    write_success = False
    stored_fidelity = 0.0
    write_attempts = 0
    if photonic_fidelity >= QuantumLink.FIDELITY_THRESHOLD:
        for attempt in range(max(int(physical_memory_model.max_write_attempts), 1)):
            write_attempts = attempt + 1
            elapsed_steps += delay_model.memory_time
            write_success = bool(
                network.agent_generator.random() < physical_memory_model.photon_to_atom_success_probability
            )
            timeline.append(
                {
                    "event": "photon_to_atom_write_attempt",
                    "terminal": int(terminal),
                    "center": int(center),
                    "attempt": int(write_attempts),
                    "success": bool(write_success),
                    "time": float(elapsed_steps),
                }
            )
            if write_success:
                break
        if write_success:
            stored_fidelity = physical_memory_model.apply_interface_fidelity(
                photonic_fidelity,
                physical_memory_model.photon_to_atom_fidelity,
            )
        else:
            failure_reason = "photon_to_atom_write_failed"
    elif failure_reason is None:
        failure_reason = "dynamic_photonic_fidelity_below_threshold"

    stored = StoredAtomicEntanglement(
        terminal=int(terminal),
        center=int(center),
        stored_at_time=float(elapsed_steps),
        initial_fidelity=float(stored_fidelity),
        write_success=bool(write_success),
        write_attempts=int(write_attempts),
        decayed_fidelity=float(stored_fidelity),
        failure_reason=failure_reason,
    )
    timeline.append(
        {
            "event": "photon_to_atom_write",
            "terminal": int(terminal),
            "center": int(center),
            "photonic_fidelity": float(photonic_fidelity),
            "write_success": bool(write_success),
            "write_attempts": int(write_attempts),
            "stored_fidelity": float(stored_fidelity),
            "time": float(elapsed_steps),
        }
    )

    path_estimate = estimate_path_from_nodes(network, terminal, center, best_candidate_path, delay_model)
    path_estimate.quantum_generation_time += float(elapsed_steps)
    path_estimate.total_time += float(elapsed_steps)
    path_estimate.fidelity = float(stored_fidelity)
    path_estimate.available_links = int(consumed_links)
    path_estimate.physical_metadata = {
        "backend": "dynamic_physical",
        "photonic_fidelity": float(photonic_fidelity),
        "stored_entanglement": stored.to_dict(),
        "timeline": timeline,
        "failure_reason": failure_reason,
        "consumed_links": int(consumed_links),
        "photonic_route_attempts": int(route_attempts),
        "photonic_candidate_paths": [[int(node) for node in path] for path in candidate_paths],
    }
    return path_estimate


def execute_dynamic_ghz_distribution_path(
    network,
    center: int,
    terminal: int,
    delay_model: DelayModel,
    physical_memory_model: PhysicalMemoryModel,
    routing_policy=None,
    max_hops: Optional[int] = None,
    device=None,
    deterministic: bool = True,
    routing_cache_static_encoding: bool = False,
    routing_prefer_progress_actions: bool = False,
) -> PathEstimate:
    """Distribute one prebuilt GHZ particle from the center to a terminal."""

    candidate_paths = _dynamic_route_path_candidates(
        network,
        center,
        terminal,
        delay_model,
        candidate_count=max(int(physical_memory_model.max_photonic_route_attempts), 1),
        routing_policy=routing_policy,
        max_hops=max_hops,
        device=device,
        deterministic=deterministic,
        cache_static_encoding=routing_cache_static_encoding,
        prefer_progress_actions=routing_prefer_progress_actions,
    )
    max_steps = int(max_hops or max(network.ttl, network.n_nodes_connected, 1))
    elapsed_steps = 0
    timeline: List[Dict[str, object]] = []
    consumed_links = 0
    best_path: List[int] = candidate_paths[0] if candidate_paths else [int(center)]
    best_fidelity = 0.0
    failure_reason = None
    route_attempts = 0

    if not candidate_paths:
        failure_reason = "ghz_distribution_route_did_not_reach_terminal"

    if failure_reason is None:
        for attempt in range(max(int(physical_memory_model.max_photonic_route_attempts), 1)):
            path = candidate_paths[min(attempt, len(candidate_paths) - 1)]
            route_attempts = attempt + 1
            attempt_elapsed_steps = 0
            attempt_link: Optional[QuantumLink] = None
            attempt_failure = None
            timeline.append(
                {
                    "event": "ghz_particle_distribution_attempt_start",
                    "center": int(center),
                    "terminal": int(terminal),
                    "attempt": int(route_attempts),
                    "path": [int(node) for node in path],
                    "time": float(elapsed_steps),
                }
            )
            for hop_index, (start, end) in enumerate(zip(path, path[1:])):
                edge = _edge_for_nodes(network, start, end)
                waited_steps = 0
                while _edge_available_links(edge) <= 0 and attempt_elapsed_steps < max_steps:
                    network.pre_step()
                    network.step()
                    waited_steps += 1
                    elapsed_steps += 1
                    attempt_elapsed_steps += 1
                if _edge_available_links(edge) <= 0:
                    attempt_failure = "dynamic_link_generation_timeout"
                    timeline.append(
                        {
                            "event": "link_generation_timeout",
                            "edge": [int(start), int(end)],
                            "attempt": int(route_attempts),
                            "waited_steps": waited_steps,
                            "time": float(elapsed_steps),
                        }
                    )
                    break

                selected_link = edge.links[-1]
                edge.links.remove(selected_link)
                network.total_active_quantum_links = max(network.total_active_quantum_links - 1, 0)
                consumed_links += 1
                elapsed_steps += 1
                attempt_elapsed_steps += 1
                timeline.append(
                    {
                        "event": "consume_distribution_link",
                        "edge": [int(start), int(end)],
                        "attempt": int(route_attempts),
                        "hop_index": hop_index,
                        "waited_steps": waited_steps,
                        "link_fidelity": float(selected_link.fidelity),
                        "time": float(elapsed_steps),
                    }
                )

                if attempt_link is None:
                    attempt_link = selected_link
                    continue

                swap_node = int(start)
                attempt_link = QuantumLink.swap(
                    attempt_link,
                    selected_link,
                    network.nodes[swap_node].swap_prob,
                    source=swap_node,
                    rng_generator=network.swap_generator,
                )
                timeline.append(
                    {
                        "event": "distribution_swap",
                        "node": swap_node,
                        "attempt": int(route_attempts),
                        "fidelity": 0.0 if attempt_link is None else float(attempt_link.fidelity),
                        "time": float(elapsed_steps),
                    }
                )
                if attempt_link is None or attempt_link.fidelity <= 0:
                    attempt_failure = "dynamic_swap_failed"
                    break

            if attempt_failure is None and attempt_link is not None:
                if attempt_link.fidelity >= QuantumLink.FIDELITY_THRESHOLD:
                    if attempt_link.fidelity > best_fidelity:
                        best_path = [int(node) for node in path]
                        best_fidelity = float(attempt_link.fidelity)
                    timeline.append(
                        {
                            "event": "ghz_particle_distribution_attempt_succeeded",
                            "center": int(center),
                            "terminal": int(terminal),
                            "attempt": int(route_attempts),
                            "distribution_fidelity": float(attempt_link.fidelity),
                            "best_distribution_fidelity": float(best_fidelity),
                            "time": float(elapsed_steps),
                        }
                    )
                    continue
                attempt_failure = "ghz_distribution_fidelity_below_threshold"

            failure_reason = attempt_failure or "ghz_distribution_route_failed"
            timeline.append(
                {
                    "event": "ghz_particle_distribution_attempt_failed",
                    "center": int(center),
                    "terminal": int(terminal),
                    "attempt": int(route_attempts),
                    "failure_reason": failure_reason,
                    "time": float(elapsed_steps),
                }
            )
        if best_fidelity > 0.0:
            failure_reason = None
            timeline.append(
                {
                    "event": "ghz_particle_distribution_candidate_selected",
                    "center": int(center),
                    "terminal": int(terminal),
                    "attempts": int(route_attempts),
                    "path": [int(node) for node in best_path],
                    "distribution_fidelity": float(best_fidelity),
                    "time": float(elapsed_steps),
                }
            )

    path_estimate = estimate_path_from_nodes(network, terminal, center, best_path, delay_model)
    path_estimate.quantum_generation_time += float(elapsed_steps)
    path_estimate.total_time += float(elapsed_steps)
    path_estimate.fidelity = float(best_fidelity)
    path_estimate.available_links = int(consumed_links)
    path_estimate.physical_metadata = {
        "backend": "dynamic_physical_distribution",
        "distribution_fidelity": float(best_fidelity),
        "timeline": timeline,
        "failure_reason": failure_reason,
        "consumed_links": int(consumed_links),
        "photonic_route_attempts": int(route_attempts),
        "photonic_candidate_paths": [[int(node) for node in path] for path in candidate_paths],
    }
    return path_estimate


def apply_memory_readout_before_ghz(
    network,
    plan: MultipartitePlan,
    routes: List[RouteExecution],
    physical_memory_model: PhysicalMemoryModel,
    delay_model: DelayModel,
    reward_weights: RewardWeights,
    fidelity_model: FidelityModel,
) -> Tuple[MultipartitePlan, List[RouteExecution], List[StoredAtomicEntanglement], List[Dict[str, object]], Dict[str, float]]:
    center_wait_time = max((route.total_time for route in routes), default=0.0)
    stored_entanglements: List[StoredAtomicEntanglement] = []
    timeline: List[Dict[str, object]] = []
    write_failures = 0
    readout_failures = 0
    memory_failures = 0
    total_memory_wait = 0.0
    total_decay_loss = 0.0
    max_readout_extra_time = 0.0

    path_by_terminal = {path.terminal: path for path in plan.paths}
    written_terminals: List[int] = []
    for route in routes:
        path_estimate = path_by_terminal.get(route.terminal)
        metadata = path_estimate.physical_metadata if path_estimate is not None else None
        stored_data = (metadata or {}).get("stored_entanglement")
        if stored_data and stored_data.get("write_success"):
            written_terminals.append(int(route.terminal))
    allowed_memory_terminals = set(written_terminals)
    if physical_memory_model.memory_slots_per_node > 0:
        allowed_memory_terminals = set(written_terminals[: physical_memory_model.memory_slots_per_node])

    for route in routes:
        path_estimate = path_by_terminal.get(route.terminal)
        metadata = path_estimate.physical_metadata if path_estimate is not None else None
        stored_data = (metadata or {}).get("stored_entanglement")
        route_timeline = (metadata or {}).get("timeline", [])
        if isinstance(route_timeline, list):
            timeline.extend(route_timeline)
        if not stored_data:
            continue

        stored = StoredAtomicEntanglement(**stored_data)
        if not stored.write_success:
            write_failures += 1
            route.completed = False
            route.failure_reason = stored.failure_reason or "photon_to_atom_write_failed"
            route.fidelity = 0.0
            stored_entanglements.append(stored)
            continue

        if int(route.terminal) not in allowed_memory_terminals:
            memory_failures += 1
            stored.failure_reason = "memory_slot_unavailable"
            route.completed = False
            route.failure_reason = stored.failure_reason
            route.fidelity = 0.0
            stored_entanglements.append(stored)
            timeline.append(
                {
                    "event": "memory_slot_unavailable",
                    "terminal": int(route.terminal),
                    "center": int(plan.center),
                    "memory_slots_per_node": int(physical_memory_model.memory_slots_per_node),
                    "time": float(route.total_time),
                }
            )
            continue

        wait_steps = max(center_wait_time - route.total_time, 0.0)
        wait_seconds = wait_steps * getattr(network, "delta_time", 1.0)
        decayed = physical_memory_model.apply_memory_decay(stored.initial_fidelity, wait_seconds)
        stored.memory_wait_time = float(wait_seconds)
        stored.decayed_fidelity = float(decayed)
        total_memory_wait += wait_seconds
        total_decay_loss += max(stored.initial_fidelity - decayed, 0.0)

        if decayed < fidelity_model.threshold:
            memory_failures += 1
            stored.failure_reason = "memory_decoherence_below_threshold"
            route.completed = False
            route.failure_reason = stored.failure_reason
            route.fidelity = 0.0
            stored_entanglements.append(stored)
            continue

        readout_success = False
        readout_attempts = 0
        for attempt in range(max(int(physical_memory_model.max_readout_attempts), 1)):
            readout_attempts = attempt + 1
            readout_success = bool(
                network.agent_generator.random() < physical_memory_model.atom_to_photon_success_probability
            )
            timeline.append(
                {
                    "event": "atom_to_photon_readout_attempt",
                    "terminal": int(route.terminal),
                    "center": int(plan.center),
                    "attempt": int(readout_attempts),
                    "success": bool(readout_success),
                    "time": float(center_wait_time + attempt * delay_model.memory_time),
                }
            )
            if readout_success:
                break
        stored.readout_success = readout_success
        stored.readout_attempts = int(readout_attempts)
        max_readout_extra_time = max(
            max_readout_extra_time,
            max(readout_attempts - 1, 0) * delay_model.memory_time,
        )
        if not readout_success:
            readout_failures += 1
            stored.failure_reason = "memory_readout_failed"
            route.completed = False
            route.failure_reason = stored.failure_reason
            route.fidelity = 0.0
            stored_entanglements.append(stored)
            continue

        readout_fidelity = physical_memory_model.apply_interface_fidelity(
            decayed,
            physical_memory_model.atom_to_photon_fidelity,
        )
        purified_fidelity, purification_success, purification_rounds = physical_memory_model.apply_purification(
            readout_fidelity,
            network.agent_generator,
        )
        stored.purification_success = bool(purification_success)
        stored.purification_rounds_completed = int(purification_rounds)
        if not purification_success:
            memory_failures += 1
            stored.failure_reason = "purification_failed"
            route.completed = False
            route.failure_reason = stored.failure_reason
            route.fidelity = 0.0
            stored_entanglements.append(stored)
            timeline.append(
                {
                    "event": "purification_failed",
                    "terminal": int(route.terminal),
                    "center": int(plan.center),
                    "rounds_completed": int(purification_rounds),
                    "time": float(center_wait_time),
                }
            )
            continue
        readout_fidelity = float(purified_fidelity)
        stored.readout_fidelity = float(readout_fidelity)
        if readout_fidelity < fidelity_model.threshold:
            memory_failures += 1
            stored.failure_reason = "memory_readout_fidelity_below_threshold"
            route.completed = False
            route.failure_reason = stored.failure_reason
            route.fidelity = 0.0
        else:
            stored.failure_reason = None
            route.fidelity = float(readout_fidelity)
            if path_estimate is not None:
                path_estimate.fidelity = float(readout_fidelity)

        timeline.append(
            {
                "event": "atom_to_photon_readout",
                "terminal": int(route.terminal),
                "center": int(plan.center),
                "wait_seconds": float(wait_seconds),
                "decayed_fidelity": float(decayed),
                "readout_success": bool(readout_success),
                "readout_fidelity": float(readout_fidelity if readout_success else 0.0),
                "purification_success": bool(purification_success),
                "purification_rounds_completed": int(purification_rounds),
                "time": float(center_wait_time),
            }
        )
        stored_entanglements.append(stored)

    updated_plan = _build_plan_from_paths(
        plan.terminals,
        plan.center,
        plan.center_strategy,
        plan.paths,
        delay_model,
        reward_weights,
        fidelity_model,
        ghz_establishment_mode="fusion_after_bipartite",
    )
    stats = {
        "write_failures": float(write_failures),
        "readout_failures": float(readout_failures),
        "memory_failures": float(memory_failures),
        "total_memory_wait": float(total_memory_wait),
        "total_decay_loss": float(total_decay_loss),
        "max_readout_extra_time": float(max_readout_extra_time),
    }
    return updated_plan, routes, stored_entanglements, timeline, stats


def generate_multipartite_request(
    network,
    target_count: int,
    request_id: int = 0,
    ttl: Optional[int] = None,
) -> MultipartiteRequest:
    if target_count < 1:
        raise ValueError("At least one target is required.")

    valid_nodes = [
        int(node.id)
        for node in network.nodes[: network.n_nodes_connected]
        if node.source_destination_valid and not node.virtual
    ]
    if len(valid_nodes) < target_count + 1:
        raise ValueError("Not enough valid nodes for the multipartite request.")

    source = int(valid_nodes[network.agent_generator.integers(len(valid_nodes))])
    reachable_targets = [
        node
        for node in valid_nodes
        if node != source
        and source in network.shortest_paths_weights
        and node in network.shortest_paths_weights[source]
    ]
    network.agent_generator.shuffle(reachable_targets)
    targets = reachable_targets[:target_count]

    if len(targets) < target_count:
        raise ValueError("Not enough reachable targets for the multipartite request.")

    return MultipartiteRequest(
        id=int(request_id),
        source=int(source),
        targets=[int(target) for target in targets],
        ttl=ttl,
    )


def execute_multipartite_request(
    network,
    request: MultipartiteRequest,
    center_strategy: str = "balanced",
    delay_model: Optional[DelayModel] = None,
    reward_weights: Optional[RewardWeights] = None,
    fidelity_model: Optional[FidelityModel] = None,
    ghz_establishment_mode: str = "prebuilt_ghz_distribution",
    state_carving_model: Optional[StateCarvingGHZModel] = None,
    max_hops: Optional[int] = None,
    selected_center: Optional[int] = None,
    bipartite_backend: str = "path_estimator",
    routing_policy=None,
    routing_device=None,
    routing_deterministic: bool = True,
    routing_cache_static_encoding: bool = False,
    routing_prefer_progress_actions: bool = False,
    physical_memory_model: Optional[PhysicalMemoryModel] = None,
) -> MultipartiteExecution:
    delay_model = delay_model or DelayModel()
    reward_weights = reward_weights or RewardWeights()
    fidelity_model = fidelity_model or FidelityModel()
    state_carving_model = state_carving_model or StateCarvingGHZModel.from_profile()
    physical_memory_model = physical_memory_model or PhysicalMemoryModel.from_profile()
    plan = plan_multipartite_entanglement(
        network,
        request.terminals,
        center_strategy="fixed" if selected_center is not None else center_strategy,
        delay_model=delay_model,
        reward_weights=reward_weights,
        fidelity_model=fidelity_model,
        ghz_establishment_mode=ghz_establishment_mode,
        state_carving_model=state_carving_model,
        center_candidates=None if selected_center is None else [int(selected_center)],
    )
    if bipartite_backend == "routing_policy":
        policy_paths = [
            estimate_path_with_routing_policy(
                network,
                terminal,
                plan.center,
                delay_model,
                routing_policy,
                max_hops=max_hops,
                device=routing_device,
                deterministic=routing_deterministic,
                cache_static_encoding=routing_cache_static_encoding,
                prefer_progress_actions=routing_prefer_progress_actions,
            )
            for terminal in request.terminals
        ]
        if ghz_establishment_mode == "prebuilt_ghz_distribution":
            policy_paths = _as_distribution_paths(network, policy_paths, delay_model)
        plan = _build_plan_from_paths(
            request.terminals,
            plan.center,
            plan.center_strategy,
            policy_paths,
            delay_model,
            reward_weights,
            fidelity_model,
            ghz_establishment_mode=ghz_establishment_mode,
            state_carving_model=state_carving_model,
        )
    elif bipartite_backend == "dynamic_physical":
        if ghz_establishment_mode == "prebuilt_ghz_distribution":
            dynamic_paths = [
                execute_dynamic_ghz_distribution_path(
                    network,
                    plan.center,
                    terminal,
                    delay_model,
                    physical_memory_model,
                    routing_policy=routing_policy,
                    max_hops=max_hops,
                    device=routing_device,
                    deterministic=routing_deterministic,
                    routing_cache_static_encoding=routing_cache_static_encoding,
                    routing_prefer_progress_actions=routing_prefer_progress_actions,
                )
                for terminal in request.terminals
            ]
        else:
            dynamic_paths = [
                execute_dynamic_physical_path(
                    network,
                    terminal,
                    plan.center,
                    delay_model,
                    physical_memory_model,
                    routing_policy=routing_policy,
                    max_hops=max_hops,
                    device=routing_device,
                    deterministic=routing_deterministic,
                    routing_cache_static_encoding=routing_cache_static_encoding,
                    routing_prefer_progress_actions=routing_prefer_progress_actions,
                )
                for terminal in request.terminals
            ]
        plan = _build_plan_from_paths(
            request.terminals,
            plan.center,
            plan.center_strategy,
            dynamic_paths,
            delay_model,
            reward_weights,
            fidelity_model,
            ghz_establishment_mode=ghz_establishment_mode,
            state_carving_model=state_carving_model,
        )
    elif bipartite_backend != "path_estimator":
        raise ValueError(f"Unknown bipartite_backend: {bipartite_backend}")

    routes: List[RouteExecution] = []
    for path_estimate in plan.paths:
        if path_estimate.terminal == request.source:
            role = "source"
            target_index = None
        else:
            role = "target"
            target_index = request.targets.index(path_estimate.terminal)

        completed = True
        failure_reason = None
        physical_metadata = path_estimate.physical_metadata
        physical_failure = (physical_metadata or {}).get("failure_reason")
        if max_hops is not None and path_estimate.hop_count > max_hops:
            completed = False
            failure_reason = "route_exceeds_max_hops"
        elif physical_failure is not None:
            completed = False
            failure_reason = str(physical_failure)
        elif path_estimate.fidelity < fidelity_model.threshold:
            completed = False
            failure_reason = "route_fidelity_below_threshold"
        elif bipartite_backend != "dynamic_physical" and not _path_has_quantum_resources(network, path_estimate.path):
            completed = False
            failure_reason = "missing_quantum_link_resource"

        routes.append(
            RouteExecution(
                role=role,
                terminal=path_estimate.terminal,
                target_index=target_index,
                path=path_estimate.path,
                hop_count=path_estimate.hop_count,
                total_time=path_estimate.total_time,
                fidelity=path_estimate.fidelity,
                completed=completed,
                failure_reason=failure_reason,
                physical_metadata=physical_metadata,
            )
        )

    center_wait_time = max((route.total_time for route in routes), default=0.0)
    physical_timeline: List[Dict[str, object]] = []
    stored_entanglements: List[StoredAtomicEntanglement] = []
    memory_wait_time = 0.0
    memory_decay_loss = 0.0
    write_success = None
    readout_success = None
    conversion_failure_rate = 0.0
    memory_failure_rate = 0.0
    readout_extra_time = 0.0
    if bipartite_backend == "dynamic_physical" and ghz_establishment_mode == "fusion_after_bipartite":
        plan, routes, stored_entanglements, physical_timeline, physical_stats = apply_memory_readout_before_ghz(
            network,
            plan,
            routes,
            physical_memory_model,
            delay_model,
            reward_weights,
            fidelity_model,
        )
        center_wait_time = max((route.total_time for route in routes), default=0.0)
        route_count_for_stats = max(len(routes), 1)
        write_failures = physical_stats["write_failures"]
        readout_failures = physical_stats["readout_failures"]
        memory_failures = physical_stats["memory_failures"]
        write_success = write_failures == 0
        readout_success = readout_failures == 0
        conversion_failure_rate = float((write_failures + readout_failures) / route_count_for_stats)
        memory_failure_rate = float(memory_failures / route_count_for_stats)
        memory_wait_time = physical_stats["total_memory_wait"]
        memory_decay_loss = physical_stats["total_decay_loss"]
        readout_extra_time = physical_stats.get("max_readout_extra_time", 0.0)
    elif bipartite_backend == "dynamic_physical":
        for route in routes:
            route_timeline = (route.physical_metadata or {}).get("timeline", [])
            if isinstance(route_timeline, list):
                physical_timeline.extend(route_timeline)

    all_routes_completed = all(route.completed for route in routes)
    if ghz_establishment_mode == "prebuilt_ghz_distribution":
        route_fidelities = [route.fidelity for route in routes]
        local_ghz_fidelity = state_carving_model.ghz_fidelity(len(request.terminals))
        ghz_input_fidelity = float(local_ghz_fidelity)
        ghz_fidelity = float(local_ghz_fidelity * np.prod(route_fidelities)) if route_fidelities and all_routes_completed else 0.0
        if ghz_fidelity < fidelity_model.threshold:
            ghz_fidelity = 0.0
        plan.ghz_input_fidelity = float(ghz_input_fidelity)
        plan.ghz_fidelity = float(ghz_fidelity)
        plan.bottleneck_fidelity = float(min(route_fidelities, default=0.0))
        local_ghz_sampled_probability = (
            float(state_carving_model.ghz_success_probability(len(request.terminals)))
            if all_routes_completed and ghz_fidelity >= fidelity_model.threshold
            else 0.0
        )
        plan.success_probability = local_ghz_sampled_probability
        total_time = center_wait_time + float(state_carving_model.generation_time)
        ghz_fusion_time = 0.0
        fusion_sampled_probability = 0.0
        fusion_succeeded = True
        local_ghz_succeeded = False
        if local_ghz_sampled_probability >= 1.0:
            local_ghz_succeeded = True
        elif local_ghz_sampled_probability > 0.0:
            local_ghz_succeeded = bool(network.agent_generator.random() < local_ghz_sampled_probability)
        success_probability = local_ghz_sampled_probability if all_routes_completed else 0.0
        success = all_routes_completed and ghz_fidelity >= fidelity_model.threshold and local_ghz_succeeded
    else:
        total_time = center_wait_time + readout_extra_time + delay_model.ghz_fusion_time
        ghz_fusion_time = float(delay_model.ghz_fusion_time)
        local_ghz_sampled_probability = 1.0
        local_ghz_succeeded = True

    if ghz_establishment_mode == "fusion_after_bipartite" and bipartite_backend == "dynamic_physical":
        route_fidelities = [route.fidelity for route in routes]
        ghz_input_fidelity, ghz_fidelity = estimate_ghz_fidelity(route_fidelities, fidelity_model) if all_routes_completed else (0.0, 0.0)
        plan.ghz_input_fidelity = float(ghz_input_fidelity)
        plan.ghz_fidelity = float(ghz_fidelity)
        plan.bottleneck_fidelity = float(min(route_fidelities, default=0.0))
        plan.success_probability = (
            float(fidelity_model.ghz_fusion_success_probability)
            if all_routes_completed and ghz_fidelity >= fidelity_model.threshold
            else 0.0
        )
        success_probability = plan.success_probability if all_routes_completed else 0.0
        fusion_sampled_probability = (
            fidelity_model.ghz_fusion_success_probability
            if all_routes_completed and ghz_fidelity >= fidelity_model.threshold
            else 0.0
        )
        fusion_succeeded = False
        if fusion_sampled_probability >= 1.0:
            fusion_succeeded = True
        elif fusion_sampled_probability > 0.0:
            fusion_succeeded = bool(network.agent_generator.random() < fusion_sampled_probability)
        success = all_routes_completed and ghz_fidelity >= fidelity_model.threshold and fusion_succeeded
    elif ghz_establishment_mode == "fusion_after_bipartite":
        ghz_input_fidelity = plan.ghz_input_fidelity if all_routes_completed else 0.0
        ghz_fidelity = plan.ghz_fidelity if all_routes_completed else 0.0
        success_probability = plan.success_probability if all_routes_completed else 0.0
        fusion_sampled_probability = (
            fidelity_model.ghz_fusion_success_probability
            if all_routes_completed and ghz_fidelity >= fidelity_model.threshold
            else 0.0
        )
        fusion_succeeded = False
        if fusion_sampled_probability >= 1.0:
            fusion_succeeded = True
        elif fusion_sampled_probability > 0.0:
            fusion_succeeded = bool(network.agent_generator.random() < fusion_sampled_probability)
        success = all_routes_completed and ghz_fidelity >= fidelity_model.threshold and fusion_succeeded

    failure_reason = None
    if not all_routes_completed:
        failed_routes = [route for route in routes if not route.completed]
        failure_reason = ",".join(sorted({route.failure_reason for route in failed_routes if route.failure_reason}))
    elif not success:
        if ghz_establishment_mode == "prebuilt_ghz_distribution":
            if ghz_fidelity < fidelity_model.threshold:
                failure_reason = "ghz_distribution_fidelity_below_threshold"
            elif local_ghz_sampled_probability <= 0.0:
                failure_reason = "state_carving_success_probability_zero"
            else:
                failure_reason = "state_carving_ghz_failed"
        elif ghz_fidelity < fidelity_model.threshold:
            failure_reason = "ghz_fidelity_below_threshold"
        elif fusion_sampled_probability <= 0.0:
            failure_reason = "ghz_fusion_success_probability_zero"
        else:
            failure_reason = "ghz_fusion_failed"

    return MultipartiteExecution(
        request=request,
        plan=plan,
        routes=routes,
        center_wait_time=float(center_wait_time),
        ghz_fusion_time=float(ghz_fusion_time),
        total_time=float(total_time),
        ghz_input_fidelity=float(ghz_input_fidelity),
        ghz_fidelity=float(ghz_fidelity),
        success_probability=float(success_probability),
        fusion_sampled_probability=float(fusion_sampled_probability),
        fusion_succeeded=bool(fusion_succeeded),
        success=bool(success),
        failure_reason=failure_reason,
        bipartite_backend=bipartite_backend,
        ghz_establishment_mode=ghz_establishment_mode,
        local_ghz_sampled_probability=float(local_ghz_sampled_probability),
        local_ghz_succeeded=bool(local_ghz_succeeded),
        physical_timeline=physical_timeline,
        stored_entanglements=stored_entanglements,
        memory_wait_time=float(memory_wait_time),
        memory_decay_loss=float(memory_decay_loss),
        write_success=write_success,
        readout_success=readout_success,
        conversion_failure_rate=float(conversion_failure_rate),
        memory_failure_rate=float(memory_failure_rate),
    )


def build_gnn_feature_package(
    network,
    terminals: Sequence[int] = (),
    center: Optional[int] = None,
) -> Dict[str, object]:
    terminal_set = set(int(node) for node in terminals)
    node_feature_names = [
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
        "terminal_reachability_ratio",
        "mean_terminal_hops_ratio",
        "max_terminal_hops_ratio",
        "min_terminal_path_fidelity",
        "mean_terminal_path_fidelity",
        "memory_wait_risk_ratio",
    ]
    edge_feature_names = [
        "available_link_ratio",
        "mean_fidelity",
        "max_fidelity",
        "distance_ratio",
        "entanglement_success_probability",
        "reservation_ratio",
        "is_alive",
    ]
    max_distance = max((network.get_node_distance(network.nodes[e.start], network.nodes[e.end]) for e in network.edges), default=1.0)
    max_storage = max(network.n_quantum_links * max(network.neighbor_count, 1), 1)

    node_features = []
    for node in network.nodes:
        edge_fidelities = []
        available_links = 0
        for edge_id in node.edges:
            edge = network.edges[edge_id]
            available_links += _edge_available_links(edge)
            edge_fidelities.extend(_edge_fidelities(edge))
        mean_fidelity = float(np.mean(edge_fidelities)) if edge_fidelities else 0.0
        distance_to_center = 0.0
        if center is not None and node.id in network.shortest_paths_weights and center in network.shortest_paths_weights[node.id]:
            distance_to_center = network.shortest_paths_weights[node.id][center] / max(network.diameter, 1)
        terminal_hops = []
        terminal_fidelities = []
        for terminal in terminal_set:
            if node.id == terminal:
                terminal_hops.append(0.0)
                terminal_fidelities.append(1.0)
                continue
            if node.id in network.shortest_paths_weights and terminal in network.shortest_paths_weights[node.id]:
                try:
                    terminal_path = nx.shortest_path(
                        network.G,
                        int(node.id),
                        int(terminal),
                        weight=network.get_edge_weight_binary,
                    )
                except nx.NetworkXNoPath:
                    continue
                terminal_hops.append(float(max(len(terminal_path) - 1, 0)))
                terminal_fidelities.append(float(estimate_path_fidelity(network, terminal_path)))
        reachability_ratio = (
            len(terminal_hops) / max(len(terminal_set), 1)
            if terminal_set
            else 0.0
        )
        mean_terminal_hops = float(np.mean(terminal_hops)) if terminal_hops else float(network.diameter)
        max_terminal_hops = float(np.max(terminal_hops)) if terminal_hops else float(network.diameter)
        min_terminal_fidelity = float(np.min(terminal_fidelities)) if terminal_fidelities else 0.0
        mean_terminal_fidelity = float(np.mean(terminal_fidelities)) if terminal_fidelities else 0.0
        memory_wait_risk = (
            (max_terminal_hops - float(np.min(terminal_hops))) / max(network.diameter, 1)
            if terminal_hops
            else 1.0
        )
        node_features.append(
            [
                node.swap_prob,
                len(node.neighbors) / max(network.neighbor_count, 1),
                available_links / max_storage,
                mean_fidelity,
                node.n_decoupling_pulses / 1024.0,
                1.0 if node.id in terminal_set else 0.0,
                1.0 if center is not None and node.id == center else 0.0,
                distance_to_center,
                1.0 if node.virtual else 0.0,
                1.0 if node.source_destination_valid else 0.0,
                float(reachability_ratio),
                mean_terminal_hops / max(network.diameter, 1),
                max_terminal_hops / max(network.diameter, 1),
                min_terminal_fidelity,
                mean_terminal_fidelity,
                float(memory_wait_risk),
            ]
        )

    edge_index = []
    edge_features = []
    for edge in network.edges:
        distance = network.get_node_distance(network.nodes[edge.start], network.nodes[edge.end])
        fidelities = _edge_fidelities(edge)
        mean_fidelity = float(np.mean(fidelities)) if fidelities else 0.0
        max_fidelity = max(fidelities, default=0.0)
        available_links = _edge_available_links(edge)
        probabilities = network.get_link_entanglement_probability(
            edge.start,
            edge.end,
            edge.intermediate_routers,
            free_slots=1,
        )
        success_probability = 1.0 - probabilities[0]
        features = [
            available_links / max(network.n_quantum_links, 1),
            mean_fidelity,
            max_fidelity,
            distance / max(max_distance, 1e-9),
            success_probability,
            edge.get_total_reservations() / max(network.n_quantum_links, 1),
            0.0 if edge.dead else 1.0,
        ]
        edge_index.append([edge.start, edge.end])
        edge_features.append(features)
        edge_index.append([edge.end, edge.start])
        edge_features.append(features)

    return {
        "node_features": np.array(node_features, dtype=np.float32),
        "edge_index": np.array(edge_index, dtype=np.int64).T if edge_index else np.zeros((2, 0), dtype=np.int64),
        "edge_features": (
            np.array(edge_features, dtype=np.float32)
            if edge_features
            else np.zeros((0, len(edge_feature_names)), dtype=np.float32)
        ),
        "node_feature_names": node_feature_names,
        "edge_feature_names": edge_feature_names,
    }
