#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from create_env import build_delay_model, build_parser, build_reward_weights, create_quantum_env
from env.gnn_encoder import GraphBatch
from env.multipartite import _edge_best_fidelity, _edge_generation_time, estimate_path_from_nodes
from env.quantum_network import QuantumLink
from env.routing_policy import (
    BipartiteRoutingPolicy,
    action_for_next_node,
    action_neighbor_nodes,
    build_bipartite_feature_package,
    valid_routing_action_mask,
)


def add_routing_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.set_defaults(n_router=30, ttl=12, reward_latency_weight=0.01, reward_resource_weight=0.02)
    parser.add_argument("--imitation-samples", type=int, default=3000)
    parser.add_argument("--imitation-batch-size", type=int, default=32)
    parser.add_argument("--rl-episodes", type=int, default=500)
    parser.add_argument(
        "--routing-teacher",
        choices=["shortest_path", "max_fidelity", "balanced"],
        default="shortest_path",
        help="Legacy single teacher used when --routing-teachers is empty.",
    )
    parser.add_argument(
        "--routing-teachers",
        type=str,
        default="shortest_path,max_fidelity,balanced",
        help="Comma-separated teachers. The highest reward teacher is used per imitation state.",
    )
    parser.add_argument(
        "--allow-wait-action",
        action="store_true",
        help="Allow action 0 to wait only when no legal next-hop action is available.",
    )
    parser.add_argument("--unreached-penalty", type=float, default=1.0, help="Penalty when a rollout does not reach target.")
    parser.add_argument(
        "--prefer-progress-actions",
        action="store_true",
        help="Mask next-hop actions to neighbors closer to target when such neighbors exist.",
    )
    parser.add_argument(
        "--curriculum-n-router",
        type=str,
        default="",
        help="Comma-separated node counts, e.g. 30,50,100. Empty keeps --n-router only.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--graph-output-dim", type=int, default=64)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--baseline-momentum", type=float, default=0.9)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=Path, default=Path("output/experiments/bipartite_routing_30.pt"))
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def valid_route_nodes(network) -> list[int]:
    return [
        int(node.id)
        for node in network.nodes[: network.n_nodes_connected]
        if node.source_destination_valid and not node.virtual
    ]


def sample_pair(network, rng: np.random.Generator) -> tuple[int, int]:
    nodes = valid_route_nodes(network)
    for _ in range(256):
        source = int(nodes[rng.integers(len(nodes))])
        reachable = [
            node
            for node in nodes
            if node != source
            and source in network.shortest_paths_weights
            and node in network.shortest_paths_weights[source]
        ]
        if reachable:
            return source, int(reachable[rng.integers(len(reachable))])
    raise ValueError("No reachable source-target pair found.")


def teacher_path(network, current: int, target: int, teacher: str) -> Optional[list[int]]:
    if current == target:
        return [int(current)]
    try:
        if teacher == "shortest_path":
            path = nx.shortest_path(network.G, current, target, weight=network.get_edge_weight_binary)
        elif teacher == "max_fidelity":
            def fidelity_weight(start, end, _attrs):
                edge = network.edge_node_association[(min(start, end), max(start, end))]
                return 1.0 / max(_edge_best_fidelity(edge, network.initial_fidelity), 1e-9)

            path = nx.shortest_path(network.G, current, target, weight=fidelity_weight)
        elif teacher == "balanced":
            def balanced_weight(start, end, _attrs):
                edge = network.edge_node_association[(min(start, end), max(start, end))]
                fidelity_penalty = 1.0 / max(_edge_best_fidelity(edge, network.initial_fidelity), 1e-9)
                generation_penalty = _edge_generation_time(network, edge)
                return 1.0 + 0.2 * fidelity_penalty + 0.1 * generation_penalty

            path = nx.shortest_path(network.G, current, target, weight=balanced_weight)
        else:
            raise ValueError(f"Unknown routing teacher: {teacher}")
    except nx.NetworkXNoPath:
        return None
    return [int(node) for node in path]


def teacher_label(network, current: int, target: int, teacher: str) -> Optional[int]:
    path = teacher_path(network, current, target, teacher)
    if path is None:
        return None
    if len(path) < 2:
        return 0
    return action_for_next_node(network, current, path[1])


def parse_routing_teachers(args) -> list[str]:
    teachers = [teacher.strip() for teacher in args.routing_teachers.split(",") if teacher.strip()]
    return teachers or [args.routing_teacher]


def score_teacher_path(env, path: list[int], target: int, args) -> float:
    reward, _ = route_reward(env, path, target, args)
    return reward


def best_teacher_label(env, current: int, target: int, args) -> tuple[Optional[int], str]:
    best = None
    for teacher in parse_routing_teachers(args):
        path = teacher_path(env.network, current, target, teacher)
        if path is None or len(path) < 2:
            label = 0 if current == target else None
            score = 0.0 if label == 0 else -float("inf")
        else:
            label = action_for_next_node(env.network, current, path[1])
            score = score_teacher_path(env, path, target, args)
        if label is None:
            continue
        current_candidate = (score, -len(path), label, teacher)
        if best is None or current_candidate > best:
            best = current_candidate
    if best is None:
        return None, ""
    _, _, label, teacher = best
    return label, teacher


def supervised_step(env, model, optimizer, args, device, rng: np.random.Generator, batch_index: int) -> dict:
    losses = []
    accuracies = []
    value_losses = []
    teacher_counts = {}
    samples = 0
    for _ in range(args.imitation_batch_size):
        source, target = sample_pair(env.network, rng)
        current = source
        visited = [current]
        max_steps = max(int(args.ttl), 1)
        for _ in range(max_steps):
            label, teacher = best_teacher_label(env, current, target, args)
            if label is None:
                break
            teacher_counts[teacher] = teacher_counts.get(teacher, 0) + 1
            features = build_bipartite_feature_package(env.network, current=current, target=target, visited=visited)
            batch = GraphBatch.from_feature_package(features, device=device)
            action_mask = valid_routing_action_mask(
                env.network,
                current,
                target,
                visited,
                device=device,
                prefer_progress=args.prefer_progress_actions,
            )
            if label >= action_mask.numel() or not bool(action_mask[label]):
                break
            teacher_path_for_value = teacher_path(env.network, current, target, teacher)
            teacher_reward = (
                score_teacher_path(env, teacher_path_for_value, target, args)
                if teacher_path_for_value is not None
                else 0.0
            )
            output = model(
                batch.node_features,
                batch.edge_index,
                batch.edge_features,
                current_node=current,
                target_node=target,
                action_neighbor_ids=action_neighbor_nodes(env.network, current, device=device),
                action_mask=action_mask,
                deterministic=False,
            )
            target_tensor = torch.as_tensor(label, dtype=torch.long, device=device)
            losses.append(F.cross_entropy(output.logits.unsqueeze(0), target_tensor.unsqueeze(0)))
            value_target = torch.as_tensor(teacher_reward, dtype=output.value.dtype, device=device)
            value_losses.append(F.mse_loss(output.value, value_target))
            accuracies.append(float(int(torch.argmax(output.logits).detach().cpu()) == label))
            samples += 1
            if label == 0:
                break
            edge = env.network.edges[env.network.get_nodes(0)[current].edges[label - 1]]
            current = int(edge.get_other_node(current))
            visited.append(current)
            if current == target:
                break

    if not losses:
        return {"phase": "imitation", "batch": batch_index, "loss": 0.0, "accuracy": 0.0, "samples": 0}

    policy_loss = torch.stack(losses).mean()
    value_loss = torch.stack(value_losses).mean() if value_losses else torch.zeros((), device=device)
    loss = policy_loss + args.value_coef * value_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    return {
        "phase": "imitation",
        "batch": batch_index,
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "accuracy": float(np.mean(accuracies)),
        "samples": samples,
        "teacher_counts": teacher_counts,
    }


def rollout(env, model, source: int, target: int, args, device, deterministic: bool) -> tuple[list[int], list[torch.Tensor], list[torch.Tensor], int]:
    current = int(source)
    path = [current]
    visited = {current}
    log_probs = []
    entropies = []
    values = []
    wait_steps = 0
    for _ in range(max(int(args.ttl), 1)):
        if current == target:
            break
        features = build_bipartite_feature_package(env.network, current=current, target=target, visited=path)
        batch = GraphBatch.from_feature_package(features, device=device)
        action_mask = valid_routing_action_mask(
            env.network,
            current,
            target,
            path,
            device=device,
            allow_idle=args.allow_wait_action,
            prefer_progress=args.prefer_progress_actions,
        )
        output = model(
            batch.node_features,
            batch.edge_index,
            batch.edge_features,
            current_node=current,
            target_node=target,
            action_neighbor_ids=action_neighbor_nodes(env.network, current, device=device),
            action_mask=action_mask,
            deterministic=deterministic,
        )
        action = int(output.action.detach().cpu())
        log_probs.append(output.log_prob)
        entropies.append(output.entropy)
        values.append(output.value)
        if action <= 0:
            if args.allow_wait_action and current != target:
                env.network.pre_step()
                env.network.step()
                wait_steps += 1
                continue
            break
        edge_slot = action - 1
        current_node = env.network.get_nodes(0)[current]
        if edge_slot >= len(current_node.edges):
            break
        edge = env.network.edges[current_node.edges[edge_slot]]
        next_node = int(edge.get_other_node(current))
        if next_node in visited:
            break
        path.append(next_node)
        visited.add(next_node)
        current = next_node
    return path, log_probs, entropies, values, wait_steps


def route_reward(env, path: list[int], target: int, args, wait_steps: int = 0) -> tuple[float, dict]:
    delay_model = build_delay_model(args)
    reward_weights = build_reward_weights(args)
    estimate = estimate_path_from_nodes(env.network, path[0], target, path, delay_model)
    estimate.total_time += float(wait_steps)
    reached = int(path[-1] == int(target))
    success = int(reached and estimate.fidelity >= QuantumLink.FIDELITY_THRESHOLD)
    reward_fidelity = estimate.fidelity if reached else 0.0
    reward = (
        reward_weights.success_bonus * success
        + reward_weights.path_found_bonus * reached
        + reward_weights.fidelity_weight * reward_fidelity
        - reward_weights.latency_weight * estimate.total_time
        - reward_weights.resource_weight * estimate.hop_count
        - args.unreached_penalty * float(not reached)
    )
    return float(reward), {
        "success": float(success),
        "reached": float(reached),
        "fidelity": float(estimate.fidelity if reached else 0.0),
        "hops": float(estimate.hop_count),
        "time": float(estimate.total_time),
        "wait_steps": float(wait_steps),
    }


def rl_step(env, model, optimizer, args, device, rng: np.random.Generator, episode: int, baseline: Optional[float]) -> tuple[dict, float]:
    env.reset()
    source, target = sample_pair(env.network, rng)
    path, log_probs, entropies, values, wait_steps = rollout(env, model, source, target, args, device, deterministic=False)
    reward, route_metrics = route_reward(env, path, target, args, wait_steps=wait_steps)
    baseline = reward if baseline is None else args.baseline_momentum * baseline + (1.0 - args.baseline_momentum) * reward
    reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=device)
    if log_probs:
        value_tensor = torch.stack(values) if values else torch.zeros((0,), dtype=torch.float32, device=device)
        advantage_tensor = reward_tensor - value_tensor.detach()
        policy_loss = -(torch.stack(log_probs) * advantage_tensor).sum()
        value_loss = F.mse_loss(value_tensor, reward_tensor.expand_as(value_tensor)) if values else torch.zeros((), device=device)
        loss = policy_loss + args.value_coef * value_loss
        if entropies:
            loss = loss - args.entropy_coef * torch.stack(entropies).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        policy_loss_value = float(policy_loss.detach().cpu())
        value_loss_value = float(value_loss.detach().cpu())
        value_value = float(value_tensor.mean().detach().cpu()) if values else 0.0
    else:
        loss_value = 0.0
        policy_loss_value = 0.0
        value_loss_value = 0.0
        value_value = 0.0
    advantage = reward - value_value
    metrics = {
        "phase": "rl",
        "episode": episode,
        "reward": reward,
        "baseline": float(baseline),
        "advantage": float(advantage),
        "value": value_value,
        "loss": loss_value,
        "policy_loss": policy_loss_value,
        "value_loss": value_loss_value,
        "source": int(source),
        "target": int(target),
        "path": [int(node) for node in path],
        **route_metrics,
    }
    return metrics, baseline


def main() -> int:
    parser = add_routing_training_args(build_parser())
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    curriculum_nodes = [
        int(value.strip())
        for value in args.curriculum_n_router.split(",")
        if value.strip()
    ]
    if not curriculum_nodes:
        curriculum_nodes = [int(args.n_router)]
    args.n_router = curriculum_nodes[0]
    if args.n_router_connected is not None:
        args.n_router_connected = min(int(args.n_router_connected), int(args.n_router))
    env = create_quantum_env(args)
    env.reset()
    source, target = sample_pair(env.network, rng)
    features = build_bipartite_feature_package(env.network, current=source, target=target, visited=[source])
    model = BipartiteRoutingPolicy(
        node_feature_dim=features["node_features"].shape[-1],
        edge_feature_dim=features["edge_features"].shape[-1],
        max_actions=env.network.neighbor_count + 1,
        hidden_dim=args.hidden_dim,
        graph_output_dim=args.graph_output_dim,
        num_gnn_layers=args.gnn_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    metrics = []
    for stage_index, n_router in enumerate(curriculum_nodes, start=1):
        args.n_router = int(n_router)
        if args.n_router_connected is not None:
            args.n_router_connected = min(int(args.n_router_connected), int(args.n_router))
        env = create_quantum_env(args)
        env.reset()
        imitation_batches = int(np.ceil(args.imitation_samples / max(args.imitation_batch_size, 1)))
        for batch_index in range(1, imitation_batches + 1):
            entry = supervised_step(env, model, optimizer, args, device, rng, batch_index)
            entry["stage"] = stage_index
            entry["n_router"] = int(n_router)
            metrics.append(entry)
            print(
                "stage={stage} n_router={n_router} phase=imitation batch={batch} "
                "loss={loss:.4f} accuracy={accuracy:.3f} samples={samples}".format(**entry)
            )

        baseline = None
        for episode in range(1, args.rl_episodes + 1):
            entry, baseline = rl_step(env, model, optimizer, args, device, rng, episode, baseline)
            entry["stage"] = stage_index
            entry["n_router"] = int(n_router)
            metrics.append(entry)
            print(
                "stage={stage} n_router={n_router} phase=rl episode={episode} reward={reward:.3f} "
                "success={success:.0f} fidelity={fidelity:.4f} hops={hops:.0f} "
                "time={time:.3f}".format(**entry)
            )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "node_feature_dim": features["node_features"].shape[-1],
                "edge_feature_dim": features["edge_features"].shape[-1],
                "max_actions": env.network.neighbor_count + 1,
                "hidden_dim": args.hidden_dim,
                "graph_output_dim": args.graph_output_dim,
                "num_gnn_layers": args.gnn_layers,
                "dropout": args.dropout,
            },
            "training_args": vars(args),
            "metrics": metrics,
        },
        args.checkpoint,
    )
    print(f"saved_checkpoint={args.checkpoint}")

    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
