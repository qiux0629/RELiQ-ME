#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from create_env import (
    build_delay_model,
    build_fidelity_model,
    build_gnn_feature_package,
    build_parser,
    build_physical_memory_model,
    build_reward_weights,
    build_state_carving_ghz_model,
    create_quantum_env,
)
from env.center_policy import MultipartiteCenterPolicy
from env.gnn_encoder import GraphBatch, adapt_feature_package_dimensions
from env.multipartite import (
    execute_multipartite_request,
    generate_multipartite_request,
    valid_center_candidates,
)
from env.routing_policy import BipartiteRoutingPolicy
from train_multipartite_center import candidate_mask_tensor, multipartite_sample_reward


def add_eval_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--policy",
        choices=["gnn", "balanced", "median", "minimax", "min-latency", "max-fidelity", "random", "all"],
        default="all",
        help="Center-selection policy to evaluate.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Center policy checkpoint from train_multipartite_center.py.")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Number of multipartite center-selection evaluations.")
    parser.add_argument("--metrics-json", type=Path, default=None, help="Optional metrics JSON path.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device.")
    parser.add_argument("--eval-seed", type=int, default=0, help="Seed used to derive per-episode requests and fusion samples.")
    parser.add_argument(
        "--bipartite-backend",
        choices=["path_estimator", "routing_policy", "dynamic_physical"],
        default="path_estimator",
        help="Terminal-to-center bipartite backend used after center selection.",
    )
    parser.add_argument("--routing-checkpoint", type=Path, default=None, help="Checkpoint from train_bipartite_routing.py.")
    parser.add_argument(
        "--routing-cache-static-encoding",
        action="store_true",
        help="Reuse one routing GNN encoding per terminal-center route. Faster on large graphs, approximate because current/visited node flags are not re-encoded each hop.",
    )
    parser.add_argument(
        "--routing-prefer-progress-actions",
        action="store_true",
        help="Restrict routing policy to neighbors closer to target when available.",
    )
    parser.add_argument(
        "--include-physical-details",
        action="store_true",
        help="Include per-event physical_timeline and stored_entanglements in metrics JSON. Disabled by default to keep experiment outputs compact.",
    )
    return parser


@torch.no_grad()
def select_gnn_center(env, request, model, device: torch.device) -> int:
    features = build_gnn_feature_package(env.network, terminals=request.terminals)
    expected_node_dim = model.gnn.node_projection[0].in_features
    expected_edge_dim = model.gnn.layers[0].edge_encoder[0].in_features
    features = adapt_feature_package_dimensions(features, expected_node_dim, expected_edge_dim)
    graph_batch = GraphBatch.from_feature_package(features, device=device)
    candidate_mask = candidate_mask_tensor(env.network, request.terminals, device)
    output = model(
        graph_batch.node_features,
        graph_batch.edge_index,
        graph_batch.edge_features,
        candidate_mask=candidate_mask,
        deterministic=True,
    )
    return int(output.action.cpu())


def evaluate_one(
    env,
    args,
    policy_name: str,
    model: Optional[MultipartiteCenterPolicy],
    routing_model: Optional[BipartiteRoutingPolicy],
    device: torch.device,
    episode: int,
) -> dict:
    env.reset()
    env.network.agent_generator = np.random.default_rng(args.eval_seed + episode)
    delay_model = build_delay_model(args)
    reward_weights = build_reward_weights(args)
    fidelity_model = build_fidelity_model(args)
    state_carving_model = build_state_carving_ghz_model(args)
    physical_memory_model = build_physical_memory_model(args)
    max_hops = args.multipartite_max_hop if args.multipartite_max_hop > 0 else None
    request = generate_multipartite_request(
        env.network,
        target_count=args.multipartite_targets,
        request_id=episode,
        ttl=max_hops,
    )

    selected_center = None
    center_strategy = policy_name
    if policy_name == "gnn":
        if model is None:
            raise ValueError("GNN center evaluation requires --checkpoint.")
        selected_center = select_gnn_center(env, request, model, device)
        center_strategy = "fixed"
    elif policy_name == "random":
        candidates = valid_center_candidates(env.network, request.terminals)
        selected_center = int(candidates[env.network.agent_generator.integers(len(candidates))])
        center_strategy = "fixed"

    execution = execute_multipartite_request(
        env.network,
        request,
        center_strategy=center_strategy,
        delay_model=delay_model,
        reward_weights=reward_weights,
        fidelity_model=fidelity_model,
        ghz_establishment_mode=args.ghz_establishment_mode,
        state_carving_model=state_carving_model,
        max_hops=max_hops,
        selected_center=selected_center,
        bipartite_backend=args.bipartite_backend,
        routing_policy=routing_model,
        routing_device=device,
        routing_deterministic=True,
        routing_cache_static_encoding=args.routing_cache_static_encoding,
        routing_prefer_progress_actions=args.routing_prefer_progress_actions,
        physical_memory_model=physical_memory_model,
    )
    route_count = len(execution.routes)
    completed_routes = sum(1 for route in execution.routes if route.completed)
    route_attempts = [
        float((route.physical_metadata or {}).get("photonic_route_attempts", 0.0))
        for route in execution.routes
    ]
    entry = {
        "policy": policy_name,
        "n_router": int(args.n_router),
        "bipartite_backend": args.bipartite_backend,
        "ghz_establishment_mode": args.ghz_establishment_mode,
        "episode": episode + 1,
        "center": int(execution.plan.center),
        "success": int(execution.success),
        "route_completion_rate": float(completed_routes / route_count) if route_count > 0 else 0.0,
        "fusion_failure": int(execution.failure_reason == "ghz_fusion_failed"),
        "local_ghz_failure": int(execution.failure_reason == "state_carving_ghz_failed"),
        "local_ghz_fidelity": float(execution.plan.local_ghz_fidelity),
        "local_ghz_success_probability": float(execution.plan.local_ghz_success_probability),
        "ghz_fidelity": float(execution.ghz_fidelity),
        "total_time": float(execution.total_time),
        "total_hops": int(execution.plan.total_hops),
        "reward": multipartite_sample_reward(execution, reward_weights),
        "failure_reason": execution.failure_reason,
        "conversion_failure_rate": float(execution.conversion_failure_rate),
        "memory_failure_rate": float(execution.memory_failure_rate),
        "memory_wait_time": float(execution.memory_wait_time),
        "memory_decay_loss": float(execution.memory_decay_loss),
        "mean_photonic_route_attempts": float(np.mean(route_attempts)) if route_attempts else 0.0,
        "write_success": execution.write_success,
        "readout_success": execution.readout_success,
    }
    if args.include_physical_details:
        execution_dict = execution.to_dict()
        entry["physical_timeline"] = execution_dict["physical_timeline"]
        entry["stored_entanglements"] = execution_dict["stored_entanglements"]
    return entry


def summarize(metrics: list[dict]) -> dict:
    by_policy = {}
    for policy in sorted({entry["policy"] for entry in metrics}):
        entries = [entry for entry in metrics if entry["policy"] == policy]
        reasons = {}
        for entry in entries:
            if entry["failure_reason"] is not None:
                reasons[entry["failure_reason"]] = reasons.get(entry["failure_reason"], 0) + 1
        by_policy[policy] = {
            "episodes": len(entries),
            "success_rate": float(np.mean([entry["success"] for entry in entries])) if entries else 0.0,
            "route_completion_rate": float(np.mean([entry["route_completion_rate"] for entry in entries])) if entries else 0.0,
            "fusion_failure_rate": float(np.mean([entry["fusion_failure"] for entry in entries])) if entries else 0.0,
            "local_ghz_failure_rate": float(np.mean([entry["local_ghz_failure"] for entry in entries])) if entries else 0.0,
            "mean_local_ghz_fidelity": float(np.mean([entry["local_ghz_fidelity"] for entry in entries])) if entries else 0.0,
            "mean_local_ghz_success_probability": float(np.mean([entry["local_ghz_success_probability"] for entry in entries])) if entries else 0.0,
            "mean_ghz_fidelity": float(np.mean([entry["ghz_fidelity"] for entry in entries])) if entries else 0.0,
            "mean_total_time": float(np.mean([entry["total_time"] for entry in entries])) if entries else 0.0,
            "mean_total_hops": float(np.mean([entry["total_hops"] for entry in entries])) if entries else 0.0,
            "mean_reward": float(np.mean([entry["reward"] for entry in entries])) if entries else 0.0,
            "failure_reason_counts": reasons,
            "conversion_failure_rate": float(np.mean([entry["conversion_failure_rate"] for entry in entries])) if entries else 0.0,
            "memory_failure_rate": float(np.mean([entry["memory_failure_rate"] for entry in entries])) if entries else 0.0,
            "mean_memory_wait_time": float(np.mean([entry["memory_wait_time"] for entry in entries])) if entries else 0.0,
            "mean_memory_decay_loss": float(np.mean([entry["memory_decay_loss"] for entry in entries])) if entries else 0.0,
            "mean_photonic_route_attempts": float(np.mean([entry["mean_photonic_route_attempts"] for entry in entries])) if entries else 0.0,
        }
    return by_policy


def main() -> int:
    parser = add_eval_args(build_parser())
    args = parser.parse_args()
    device = torch.device(args.device)
    policies = ["gnn", "balanced", "median", "minimax", "min-latency", "max-fidelity", "random"] if args.policy == "all" else [args.policy]

    model = None
    if "gnn" in policies:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for policy=gnn or policy=all.")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = MultipartiteCenterPolicy(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()

    routing_model = None
    if args.bipartite_backend in {"routing_policy", "dynamic_physical"} and args.routing_checkpoint is not None:
        routing_checkpoint = torch.load(args.routing_checkpoint, map_location=device, weights_only=False)
        routing_model = BipartiteRoutingPolicy(**routing_checkpoint["model_config"]).to(device)
        routing_model.load_state_dict(routing_checkpoint["model_state_dict"], strict=False)
        routing_model.eval()
    elif args.bipartite_backend == "routing_policy":
        raise ValueError("--routing-checkpoint is required when --bipartite-backend routing_policy.")

    metrics = []
    for policy in policies:
        env = create_quantum_env(args)
        for episode in range(args.eval_episodes):
            entry = evaluate_one(env, args, policy, model, routing_model, device, episode)
            metrics.append(entry)
            print(
                "policy={policy} episode={episode} success={success} center={center} "
                "reward={reward:.3f} time={total_time:.3f} hops={total_hops}".format(**entry)
            )

    summary = summarize(metrics)
    result = {
        "summary": summary,
        "metrics": metrics,
        "bipartite_backend": args.bipartite_backend,
        "ghz_establishment_mode": args.ghz_establishment_mode,
        "state_carving_ghz_model": build_state_carving_ghz_model(args).__dict__,
        "physical_memory_model": build_physical_memory_model(args).__dict__,
    }
    print(json.dumps({"summary": summary}, indent=2))
    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
