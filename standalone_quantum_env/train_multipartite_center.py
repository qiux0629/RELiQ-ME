#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

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
from env.gnn_encoder import GraphBatch
from env.multipartite import (
    execute_multipartite_request,
    generate_multipartite_request,
    plan_multipartite_entanglement,
    valid_center_candidates,
)
from env.routing_policy import BipartiteRoutingPolicy


def add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--train-episodes", type=int, default=100, help="Number of center-selection training episodes.")
    parser.add_argument("--imitation-episodes", type=int, default=0, help="Heuristic imitation pretraining episodes before RL.")
    parser.add_argument(
        "--center-teachers",
        type=str,
        default="balanced,max-fidelity,min-latency",
        help="Comma-separated heuristic teachers; the highest-reward teacher labels each imitation sample.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Optimizer learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="GNN hidden dimension.")
    parser.add_argument("--graph-output-dim", type=int, default=128, help="Graph embedding dimension.")
    parser.add_argument("--gnn-layers", type=int, default=3, help="Number of GNN message-passing layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Model dropout.")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient.")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Actor-critic value loss coefficient.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument(
        "--bipartite-backend",
        choices=["path_estimator", "routing_policy", "dynamic_physical"],
        default="path_estimator",
        help="Backend used for RL fine-tuning episodes.",
    )
    parser.add_argument("--routing-checkpoint", type=Path, default=None, help="Routing checkpoint for routing_policy/dynamic_physical fine-tuning.")
    parser.add_argument("--routing-cache-static-encoding", action="store_true", help="Approximate large-graph routing acceleration.")
    parser.add_argument("--routing-prefer-progress-actions", action="store_true", help="Use progress-constrained routing actions during fine-tuning.")
    parser.add_argument("--checkpoint", type=Path, default=Path("output/multipartite_center_policy.pt"), help="Checkpoint output path.")
    parser.add_argument("--metrics-json", type=Path, default=None, help="Optional training metrics JSON path.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device.")
    parser.add_argument("--seed", type=int, default=0, help="Torch and NumPy seed.")
    return parser


def candidate_mask_tensor(network, terminals, device: torch.device) -> torch.Tensor:
    candidates = valid_center_candidates(network, terminals)
    mask = torch.zeros(network.n_nodes, dtype=torch.bool, device=device)
    if candidates:
        mask[torch.as_tensor(candidates, dtype=torch.long, device=device)] = True
    return mask


def multipartite_sample_reward(execution, reward_weights) -> float:
    all_routes_completed = all(route.completed for route in execution.routes)
    return float(
        reward_weights.success_bonus * float(execution.success)
        + reward_weights.path_found_bonus * float(all_routes_completed)
        + reward_weights.fidelity_weight * execution.ghz_fidelity
        - reward_weights.latency_weight * execution.total_time
        - reward_weights.resource_weight * execution.plan.total_hops
    )


def best_teacher_center(
    network,
    terminals,
    args,
    delay_model,
    reward_weights,
    fidelity_model,
    state_carving_model,
) -> tuple[int, float, str]:
    best = None
    teachers = [teacher.strip() for teacher in args.center_teachers.split(",") if teacher.strip()]
    for teacher in teachers:
        plan = plan_multipartite_entanglement(
            network,
            terminals,
            center_strategy=teacher,
            delay_model=delay_model,
            reward_weights=reward_weights,
            fidelity_model=fidelity_model,
            ghz_establishment_mode=args.ghz_establishment_mode,
            state_carving_model=state_carving_model,
        )
        current = (float(plan.reward), int(plan.center), teacher)
        if best is None or current > best:
            best = current
    if best is None:
        raise ValueError("No center teacher strategies configured.")
    reward, center, teacher = best
    return center, reward, teacher


def imitation_episode(env, model, optimizer, args, device, episode: int) -> Dict[str, float]:
    env.reset()
    env.network.agent_generator = np.random.default_rng(args.seed + 100000 + episode)
    delay_model = build_delay_model(args)
    reward_weights = build_reward_weights(args)
    fidelity_model = build_fidelity_model(args)
    state_carving_model = build_state_carving_ghz_model(args)
    max_hops = args.multipartite_max_hop if args.multipartite_max_hop > 0 else None
    request = generate_multipartite_request(
        env.network,
        target_count=args.multipartite_targets,
        request_id=episode,
        ttl=max_hops,
    )
    label_center, teacher_reward, teacher_name = best_teacher_center(
        env.network,
        request.terminals,
        args,
        delay_model,
        reward_weights,
        fidelity_model,
        state_carving_model,
    )
    features = build_gnn_feature_package(env.network, terminals=request.terminals)
    graph_batch = GraphBatch.from_feature_package(features, device=device)
    candidate_mask = candidate_mask_tensor(env.network, request.terminals, device)
    output = model(
        graph_batch.node_features,
        graph_batch.edge_index,
        graph_batch.edge_features,
        candidate_mask=candidate_mask,
    )
    label = torch.as_tensor(label_center, dtype=torch.long, device=device)
    policy_loss = F.cross_entropy(output.logits.unsqueeze(0), label.unsqueeze(0))
    value_target = torch.as_tensor([teacher_reward], dtype=output.value.dtype, device=device)
    value_loss = F.mse_loss(output.value.view(-1), value_target)
    loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * output.entropy

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prediction = int(torch.argmax(output.logits.detach()).cpu())
    return {
        "phase": "imitation",
        "episode": episode,
        "teacher": teacher_name,
        "teacher_center": int(label_center),
        "predicted_center": prediction,
        "teacher_reward": float(teacher_reward),
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "accuracy": float(prediction == int(label_center)),
        "entropy": float(output.entropy.detach().cpu()),
    }


def train_episode(env, model, optimizer, args, device, routing_model=None) -> Dict[str, float]:
    env.reset()
    env.network.agent_generator = np.random.default_rng(args.seed + train_episode.counter)
    train_episode.counter += 1

    delay_model = build_delay_model(args)
    reward_weights = build_reward_weights(args)
    fidelity_model = build_fidelity_model(args)
    state_carving_model = build_state_carving_ghz_model(args)
    physical_memory_model = build_physical_memory_model(args)
    max_hops = args.multipartite_max_hop if args.multipartite_max_hop > 0 else None

    request = generate_multipartite_request(
        env.network,
        target_count=args.multipartite_targets,
        request_id=train_episode.counter,
        ttl=max_hops,
    )
    features = build_gnn_feature_package(env.network, terminals=request.terminals)
    graph_batch = GraphBatch.from_feature_package(features, device=device)
    candidate_mask = candidate_mask_tensor(env.network, request.terminals, device)
    if not bool(candidate_mask.any()):
        raise ValueError("No valid center candidates for generated multipartite request.")

    output = model(
        graph_batch.node_features,
        graph_batch.edge_index,
        graph_batch.edge_features,
        candidate_mask=candidate_mask,
    )
    selected_center = int(output.action.detach().cpu())
    execution = execute_multipartite_request(
        env.network,
        request,
        center_strategy="fixed",
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
    reward = multipartite_sample_reward(execution, reward_weights)
    reward_tensor = torch.as_tensor([reward], dtype=output.value.dtype, device=device)
    advantage = reward_tensor - output.value.detach().view(-1)
    policy_loss = -output.log_prob * advantage.squeeze(0)
    value_loss = F.mse_loss(output.value.view(-1), reward_tensor)
    loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * output.entropy

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    metrics = {
        "phase": "rl",
        "reward": reward,
        "value": float(output.value.detach().view(-1)[0].cpu()),
        "advantage": float(advantage.detach().cpu()[0]),
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(output.entropy.detach().cpu()),
        "center": selected_center,
        "success": float(execution.success),
        "fusion_succeeded": float(execution.fusion_succeeded),
        "ghz_fidelity": float(execution.ghz_fidelity),
        "total_time": float(execution.total_time),
        "total_hops": float(execution.plan.total_hops),
    }
    return metrics


train_episode.counter = 0


def main() -> int:
    parser = add_training_args(build_parser())
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    env = create_quantum_env(args)
    env.reset()
    features = build_gnn_feature_package(env.network)
    model = MultipartiteCenterPolicy(
        node_feature_dim=features["node_features"].shape[-1],
        edge_feature_dim=features["edge_features"].shape[-1],
        hidden_dim=args.hidden_dim,
        graph_output_dim=args.graph_output_dim,
        num_gnn_layers=args.gnn_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    routing_model = None
    if args.bipartite_backend in {"routing_policy", "dynamic_physical"}:
        if args.routing_checkpoint is None:
            raise ValueError("--routing-checkpoint is required for routing_policy/dynamic_physical center training.")
        routing_checkpoint = torch.load(args.routing_checkpoint, map_location=device, weights_only=False)
        routing_model = BipartiteRoutingPolicy(**routing_checkpoint["model_config"]).to(device)
        routing_model.load_state_dict(routing_checkpoint["model_state_dict"], strict=False)
        routing_model.eval()

    metrics = []
    for episode in range(1, args.imitation_episodes + 1):
        episode_metrics = imitation_episode(env, model, optimizer, args, device, episode)
        metrics.append(episode_metrics)
        print(
            "phase=imitation episode={episode} teacher={teacher} accuracy={accuracy:.0f} "
            "loss={loss:.4f} teacher_reward={teacher_reward:.3f}".format(**episode_metrics)
        )

    for episode in range(1, args.train_episodes + 1):
        episode_metrics = train_episode(env, model, optimizer, args, device, routing_model=routing_model)
        episode_metrics["episode"] = episode
        metrics.append(episode_metrics)
        print(
            "episode={episode} reward={reward:.3f} success={success:.0f} center={center} "
            "ghz_fidelity={ghz_fidelity:.4f} time={total_time:.3f} hops={total_hops:.0f}".format(**episode_metrics)
        )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "node_feature_dim": features["node_features"].shape[-1],
                "edge_feature_dim": features["edge_features"].shape[-1],
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
