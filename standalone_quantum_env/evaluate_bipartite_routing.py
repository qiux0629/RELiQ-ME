#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from create_env import build_delay_model, build_parser, build_reward_weights, create_quantum_env
from env.multipartite import _edge_best_fidelity, estimate_path_from_nodes, estimate_path_with_routing_policy
from env.quantum_network import QuantumLink
from env.routing_policy import BipartiteRoutingPolicy
from train_bipartite_routing import sample_pair


def add_eval_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--policy",
        choices=["routing_policy", "shortest_path", "max_fidelity", "random", "all"],
        default="all",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--unreached-penalty", type=float, default=1.0)
    parser.add_argument(
        "--routing-cache-static-encoding",
        action="store_true",
        help="Reuse one routing GNN encoding per route for faster large-graph evaluation.",
    )
    parser.add_argument(
        "--routing-prefer-progress-actions",
        action="store_true",
        help="Restrict routing policy to neighbors closer to target when available.",
    )
    return parser


def max_fidelity_path(network, source: int, target: int) -> list[int]:
    def weight(start, end, _attrs):
        edge = network.edge_node_association[(min(start, end), max(start, end))]
        fidelity = _edge_best_fidelity(edge, network.initial_fidelity)
        return 1.0 / max(fidelity, 1e-9)

    return [int(node) for node in nx.shortest_path(network.G, source, target, weight=weight)]


def random_path(network, source: int, target: int, max_hops: int, rng: np.random.Generator) -> list[int]:
    current = int(source)
    path = [current]
    visited = {current}
    for _ in range(max(max_hops, 1)):
        if current == int(target):
            break
        candidates = []
        for edge_id in network.get_nodes(0)[current].edges[: network.neighbor_count]:
            other = int(network.edges[edge_id].get_other_node(current))
            if other not in visited and nx.has_path(network.G, other, int(target)):
                candidates.append(other)
        if not candidates:
            break
        current = int(candidates[rng.integers(len(candidates))])
        path.append(current)
        visited.add(current)
    return path


def route_reward(env, estimate, reached: bool, args) -> float:
    reward_weights = build_reward_weights(args)
    success = reached and estimate.fidelity >= QuantumLink.FIDELITY_THRESHOLD
    reward_fidelity = estimate.fidelity if reached else 0.0
    return float(
        reward_weights.success_bonus * float(success)
        + reward_weights.path_found_bonus * float(reached)
        + reward_weights.fidelity_weight * reward_fidelity
        - reward_weights.latency_weight * estimate.total_time
        - reward_weights.resource_weight * estimate.hop_count
        - getattr(args, "unreached_penalty", 1.0) * float(not reached)
    )


def evaluate_one(env, args, policy_name: str, model, device, episode: int) -> dict:
    env.reset()
    rng = np.random.default_rng(args.eval_seed + episode)
    env.network.agent_generator = rng
    source, target = sample_pair(env.network, rng)
    max_hops = int(args.ttl) if int(args.ttl) > 0 else env.network.n_nodes_connected
    delay_model = build_delay_model(args)

    if policy_name == "routing_policy":
        if model is None:
            raise ValueError("--checkpoint is required for routing_policy evaluation.")
        estimate = estimate_path_with_routing_policy(
            env.network,
            source,
            target,
            delay_model,
            model,
            max_hops=max_hops,
            device=device,
            deterministic=True,
            cache_static_encoding=args.routing_cache_static_encoding,
            prefer_progress_actions=args.routing_prefer_progress_actions,
        )
    elif policy_name == "shortest_path":
        path = nx.shortest_path(env.network.G, source, target, weight=env.network.get_edge_weight_binary)
        estimate = estimate_path_from_nodes(env.network, source, target, path, delay_model)
    elif policy_name == "max_fidelity":
        path = max_fidelity_path(env.network, source, target)
        estimate = estimate_path_from_nodes(env.network, source, target, path, delay_model)
    elif policy_name == "random":
        path = random_path(env.network, source, target, max_hops, rng)
        estimate = estimate_path_from_nodes(env.network, source, target, path, delay_model)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    reached = estimate.path[-1] == int(target)
    success = reached and estimate.fidelity >= QuantumLink.FIDELITY_THRESHOLD
    return {
        "policy": policy_name,
        "episode": episode + 1,
        "source": int(source),
        "target": int(target),
        "path": estimate.path,
        "success": int(success),
        "reached": int(reached),
        "fidelity": float(estimate.fidelity if reached else 0.0),
        "total_time": float(estimate.total_time),
        "hops": int(estimate.hop_count),
        "reward": route_reward(env, estimate, reached, args),
    }


def summarize(metrics: list[dict]) -> dict:
    result = {}
    for policy in sorted({entry["policy"] for entry in metrics}):
        entries = [entry for entry in metrics if entry["policy"] == policy]
        result[policy] = {
            "episodes": len(entries),
            "success_rate": float(np.mean([entry["success"] for entry in entries])) if entries else 0.0,
            "reached_rate": float(np.mean([entry["reached"] for entry in entries])) if entries else 0.0,
            "mean_fidelity": float(np.mean([entry["fidelity"] for entry in entries])) if entries else 0.0,
            "mean_total_time": float(np.mean([entry["total_time"] for entry in entries])) if entries else 0.0,
            "mean_hops": float(np.mean([entry["hops"] for entry in entries])) if entries else 0.0,
            "mean_reward": float(np.mean([entry["reward"] for entry in entries])) if entries else 0.0,
        }
    return result


def main() -> int:
    parser = add_eval_args(build_parser())
    args = parser.parse_args()
    device = torch.device(args.device)
    policies = ["routing_policy", "shortest_path", "max_fidelity", "random"] if args.policy == "all" else [args.policy]

    model = None
    if "routing_policy" in policies:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for policy=routing_policy or policy=all.")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = BipartiteRoutingPolicy(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()

    metrics = []
    for policy in policies:
        env = create_quantum_env(args)
        for episode in range(args.eval_episodes):
            entry = evaluate_one(env, args, policy, model, device, episode)
            metrics.append(entry)
            print(
                "policy={policy} episode={episode} success={success} reward={reward:.3f} "
                "fidelity={fidelity:.4f} time={total_time:.3f} hops={hops}".format(**entry)
            )

    summary = summarize(metrics)
    result = {"summary": summary, "metrics": metrics}
    print(json.dumps({"summary": summary}, indent=2))
    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
