from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from env.ghz_simulator import GhzSimulatorBackend, GhzSimulationResult
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

    def to_dict(self) -> Dict[str, object]:
        return {
            "terminals": self.terminals,
            "center": self.center,
            "center_strategy": self.center_strategy,
            "total_time": self.total_time,
            "total_hops": self.total_hops,
            "bottleneck_fidelity": self.bottleneck_fidelity,
            "ghz_input_fidelity": self.ghz_input_fidelity,
            "ghz_fidelity": self.ghz_fidelity,
            "fusion_gate_fidelity": self.fusion_gate_fidelity,
            "fusion_success_probability": self.fusion_success_probability,
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
    success: bool
    failure_reason: Optional[str]
    ghz_simulation: Optional[GhzSimulationResult] = None
    rl_reward: Optional[float] = None
    selected_center_q_value: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        result = {
            "flow": [
                "one_to_many_request_generation",
                "concurrent_target_routing",
                "center_waits_for_all_paths",
                "ghz_fusion",
                "multipartite_request_success_or_failure",
            ],
            "request": self.request.to_dict(),
            "center": int(self.plan.center),
            "center_strategy": self.plan.center_strategy,
            "center_wait_time": float(self.center_wait_time),
            "ghz_fusion_time": float(self.ghz_fusion_time),
            "total_time": float(self.total_time),
            "ghz_input_fidelity": float(self.ghz_input_fidelity),
            "ghz_fidelity": float(self.ghz_fidelity),
            "success_probability": float(self.success_probability),
            "success": bool(self.success),
            "failure_reason": self.failure_reason,
            "routes": [route.to_dict() for route in self.routes],
            "plan": self.plan.to_dict(),
        }
        if self.ghz_simulation is not None:
            result["ghz_simulation"] = self.ghz_simulation.to_dict()
            result["ghz_simulator_backend"] = self.ghz_simulation.backend
        if self.rl_reward is not None:
            result["rl_reward"] = float(self.rl_reward)
        if self.selected_center_q_value is not None:
            result["selected_center_q_value"] = float(self.selected_center_q_value)
        return result


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


def estimate_path(
    network,
    terminal: int,
    center: int,
    delay_model: DelayModel,
) -> PathEstimate:
    path = nx.shortest_path(network.G, terminal, center, weight=network.get_edge_weight_binary)
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


def get_valid_center_candidates(
    network,
    terminals: Sequence[int],
    candidates: Optional[Iterable[int]] = None,
) -> List[int]:
    return _valid_centers(network, terminals, candidates)


def calculate_center_selection_reward(
    path_found: bool,
    ghz_success: bool,
    ghz_input_fidelity: float,
    ghz_fidelity: float,
) -> float:
    if not path_found:
        return 0.0
    if ghz_success:
        return float(ghz_fidelity)
    return float(0.5 * ghz_input_fidelity)


def _score_plan(
    paths: Sequence[PathEstimate],
    center: int,
    strategy: str,
    reward_weights: RewardWeights,
    fidelity_model: FidelityModel,
) -> Tuple[float, float, float, float, float, int]:
    total_time = max((path.total_time for path in paths), default=0.0)
    total_hops = sum(path.hop_count for path in paths)
    path_fidelities = [path.fidelity for path in paths]
    bottleneck_fidelity = min(path_fidelities, default=0.0)
    ghz_input_fidelity, ghz_fidelity = estimate_ghz_fidelity(path_fidelities, fidelity_model)
    success_probability = (
        fidelity_model.ghz_fusion_success_probability
        if bottleneck_fidelity >= fidelity_model.threshold and ghz_fidelity >= fidelity_model.threshold
        else 0.0
    )
    reward = (
        reward_weights.success_bonus * success_probability
        + reward_weights.path_found_bonus
        + reward_weights.fidelity_weight * ghz_fidelity
        - reward_weights.latency_weight * total_time
        - reward_weights.resource_weight * total_hops
    )

    if strategy == "median":
        score = -total_hops
    elif strategy == "minimax":
        score = -max((path.hop_count for path in paths), default=0)
    elif strategy == "max-fidelity":
        score = ghz_fidelity
    elif strategy == "balanced":
        score = reward
    elif strategy == "min-latency":
        score = -total_time
    else:
        score = reward

    return score, total_time, bottleneck_fidelity, ghz_input_fidelity, ghz_fidelity, total_hops


def _build_multipartite_plan(
    terminals: Sequence[int],
    center: int,
    center_strategy: str,
    paths: Sequence[PathEstimate],
    delay_model: DelayModel,
    reward_weights: RewardWeights,
    fidelity_model: FidelityModel,
) -> MultipartitePlan:
    total_time = max((path.total_time for path in paths), default=0.0) + delay_model.ghz_fusion_time
    total_hops = sum(path.hop_count for path in paths)
    path_fidelities = [path.fidelity for path in paths]
    bottleneck_fidelity = min(path_fidelities, default=0.0)
    ghz_input_fidelity, ghz_fidelity = estimate_ghz_fidelity(path_fidelities, fidelity_model)
    success_probability = (
        fidelity_model.ghz_fusion_success_probability
        if bottleneck_fidelity >= fidelity_model.threshold and ghz_fidelity >= fidelity_model.threshold
        else 0.0
    )
    reward = (
        reward_weights.success_bonus * success_probability
        + reward_weights.path_found_bonus
        + reward_weights.fidelity_weight * ghz_fidelity
        - reward_weights.latency_weight * total_time
        - reward_weights.resource_weight * total_hops
    )

    return MultipartitePlan(
        terminals=[int(node) for node in terminals],
        center=int(center),
        center_strategy=center_strategy,
        paths=list(paths),
        total_time=float(total_time),
        total_hops=int(total_hops),
        bottleneck_fidelity=float(bottleneck_fidelity),
        ghz_input_fidelity=float(ghz_input_fidelity),
        ghz_fidelity=float(ghz_fidelity),
        fusion_gate_fidelity=float(fidelity_model.ghz_fusion_gate_fidelity),
        fusion_success_probability=float(fidelity_model.ghz_fusion_success_probability),
        success_probability=float(success_probability),
        reward=float(reward),
    )


def plan_multipartite_entanglement_for_center(
    network,
    terminals: Sequence[int],
    center: int,
    center_strategy: str = "fixed",
    delay_model: Optional[DelayModel] = None,
    reward_weights: Optional[RewardWeights] = None,
    fidelity_model: Optional[FidelityModel] = None,
) -> MultipartitePlan:
    if len(terminals) < 2:
        raise ValueError("At least two terminals are required.")

    delay_model = delay_model or DelayModel()
    reward_weights = reward_weights or RewardWeights()
    fidelity_model = fidelity_model or FidelityModel()
    terminals = [int(node) for node in terminals]
    center = int(center)
    if center not in _valid_centers(network, terminals, [center]):
        raise ValueError(f"Selected center {center} is not reachable from all terminals.")

    paths = [estimate_path(network, terminal, center, delay_model) for terminal in terminals]
    return _build_multipartite_plan(
        terminals,
        center,
        center_strategy,
        paths,
        delay_model,
        reward_weights,
        fidelity_model,
    )


def plan_multipartite_entanglement(
    network,
    terminals: Sequence[int],
    center_strategy: str = "balanced",
    delay_model: Optional[DelayModel] = None,
    reward_weights: Optional[RewardWeights] = None,
    fidelity_model: Optional[FidelityModel] = None,
    center_candidates: Optional[Iterable[int]] = None,
) -> MultipartitePlan:
    if len(terminals) < 2:
        raise ValueError("At least two terminals are required.")

    delay_model = delay_model or DelayModel()
    reward_weights = reward_weights or RewardWeights()
    fidelity_model = fidelity_model or FidelityModel()
    terminals = [int(node) for node in terminals]
    centers = _valid_centers(network, terminals, center_candidates)
    if not centers:
        raise ValueError("No reachable center candidate found for terminals.")
    if center_strategy == "rl":
        raise ValueError("RL center selection requires an explicit center from a policy checkpoint.")

    if center_strategy == "random":
        center = int(centers[network.topology_generator.integers(len(centers))])
        paths = [estimate_path(network, terminal, center, delay_model) for terminal in terminals]
    else:
        best = None
        for candidate in centers:
            paths = [estimate_path(network, terminal, candidate, delay_model) for terminal in terminals]
            score, total_time, bottleneck_fidelity, ghz_input_fidelity, ghz_fidelity, total_hops = _score_plan(
                paths,
                candidate,
                center_strategy,
                reward_weights,
                fidelity_model,
            )
            tie_breaker = (-total_time, ghz_fidelity, -total_hops, -candidate)
            current = (score, tie_breaker, candidate, paths, total_time, bottleneck_fidelity, ghz_input_fidelity, ghz_fidelity, total_hops)
            if best is None or current[:2] > best[:2]:
                best = current
        _, _, center, paths, _, _, _, _, _ = best

    return _build_multipartite_plan(
        terminals=terminals,
        center=center,
        center_strategy=center_strategy,
        paths=paths,
        delay_model=delay_model,
        reward_weights=reward_weights,
        fidelity_model=fidelity_model,
    )


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
    max_hops: Optional[int] = None,
    center: Optional[int] = None,
    ghz_simulator: Optional[GhzSimulatorBackend] = None,
    use_rl_reward: bool = False,
    selected_center_q_value: Optional[float] = None,
) -> MultipartiteExecution:
    delay_model = delay_model or DelayModel()
    reward_weights = reward_weights or RewardWeights()
    fidelity_model = fidelity_model or FidelityModel()
    if center is None:
        plan = plan_multipartite_entanglement(
            network,
            request.terminals,
            center_strategy=center_strategy,
            delay_model=delay_model,
            reward_weights=reward_weights,
            fidelity_model=fidelity_model,
        )
    else:
        plan = plan_multipartite_entanglement_for_center(
            network,
            request.terminals,
            center=center,
            center_strategy=center_strategy,
            delay_model=delay_model,
            reward_weights=reward_weights,
            fidelity_model=fidelity_model,
        )

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
        if max_hops is not None and path_estimate.hop_count > max_hops:
            completed = False
            failure_reason = "route_exceeds_max_hops"
        elif path_estimate.fidelity < fidelity_model.threshold:
            completed = False
            failure_reason = "route_fidelity_below_threshold"
        elif not _path_has_quantum_resources(network, path_estimate.path):
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
            )
        )

    center_wait_time = max((route.total_time for route in routes), default=0.0)
    total_time = center_wait_time + delay_model.ghz_fusion_time
    all_routes_completed = all(route.completed for route in routes)
    ghz_simulation = None
    if all_routes_completed and ghz_simulator is not None:
        ghz_simulation = ghz_simulator.simulate(
            [path.fidelity for path in plan.paths],
            fidelity_model,
        )
        ghz_input_fidelity = ghz_simulation.ghz_input_fidelity
        ghz_fidelity = ghz_simulation.ghz_fidelity
        success_probability = ghz_simulation.success_probability
        success = ghz_simulation.success
    else:
        ghz_input_fidelity = plan.ghz_input_fidelity if all_routes_completed else 0.0
        ghz_fidelity = plan.ghz_fidelity if all_routes_completed else 0.0
        success_probability = plan.success_probability if all_routes_completed else 0.0
        success = all_routes_completed and ghz_fidelity >= fidelity_model.threshold and success_probability > 0

    failure_reason = None
    if not all_routes_completed:
        failed_routes = [route for route in routes if not route.completed]
        failure_reason = ",".join(sorted({route.failure_reason for route in failed_routes if route.failure_reason}))
    elif not success:
        if ghz_fidelity < fidelity_model.threshold:
            failure_reason = "ghz_fidelity_below_threshold"
        else:
            failure_reason = "ghz_fusion_success_probability_zero"
    rl_reward = (
        calculate_center_selection_reward(
            path_found=True,
            ghz_success=success,
            ghz_input_fidelity=plan.ghz_input_fidelity,
            ghz_fidelity=ghz_fidelity,
        )
        if use_rl_reward
        else None
    )

    return MultipartiteExecution(
        request=request,
        plan=plan,
        routes=routes,
        center_wait_time=float(center_wait_time),
        ghz_fusion_time=float(delay_model.ghz_fusion_time),
        total_time=float(total_time),
        ghz_input_fidelity=float(ghz_input_fidelity),
        ghz_fidelity=float(ghz_fidelity),
        success_probability=float(success_probability),
        success=bool(success),
        failure_reason=failure_reason,
        ghz_simulation=ghz_simulation,
        rl_reward=rl_reward,
        selected_center_q_value=selected_center_q_value,
    )


def build_gnn_feature_package(
    network,
    terminals: Sequence[int] = (),
    center: Optional[int] = None,
) -> Dict[str, object]:
    terminal_set = set(int(node) for node in terminals)
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
        "edge_features": np.array(edge_features, dtype=np.float32),
        "node_feature_names": [
            "swap_probability",
            "degree_ratio",
            "storage_utilization_ratio",
            "mean_incident_fidelity",
            "decoupling_pulses_ratio",
            "is_terminal",
            "is_center",
            "distance_to_center_ratio",
        ],
        "edge_feature_names": [
            "available_link_ratio",
            "mean_fidelity",
            "max_fidelity",
            "distance_ratio",
            "entanglement_success_probability",
            "reservation_ratio",
            "is_alive",
        ],
    }
