#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import sys
from pathlib import Path

import torch


BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

from create_env import build_parser as build_env_parser
from create_env import build_delay_model, build_fidelity_model, build_reward_weights, create_quantum_env
from env.ghz_simulator import create_ghz_simulator
from env.multipartite import generate_multipartite_request
from env.multipartite_rl import (
    CenterDqnPolicy,
    CenterSelectionEnv,
    ReplayBuffer,
    Transition,
    compute_one_step_dqn_loss,
    save_policy_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a DQN center selector for multipartite entanglement.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes.")
    parser.add_argument("--checkpoint", type=Path, default=Path("output/multipartite_rl.pt"), help="Checkpoint path.")
    parser.add_argument("--n-router", type=int, default=30, help="Number of quantum network nodes.")
    parser.add_argument("--n-data", type=int, default=2, help="Number of legacy RELiQ requests created during reset.")
    parser.add_argument("--multipartite-targets", type=int, default=3, help="Number of one-to-many request targets.")
    parser.add_argument("--multipartite-max-hop", type=int, default=0, help="Maximum route hops. 0 disables the check.")
    parser.add_argument("--ghz-simulator", choices=["auto", "qpanda", "numpy"], default="auto", help="GHZ simulator backend.")
    parser.add_argument("--ghz-shots", type=int, default=1024, help="Shots used by the GHZ simulator.")
    parser.add_argument("--ghz-gate-noise", type=float, default=0.0, help="Depolarizing noise probability for QPanda3 H/CNOT gates.")
    parser.add_argument("--ghz-readout-error", type=float, default=0.0, help="Readout bit-flip error probability for QPanda3.")
    parser.add_argument("--seed", type=int, default=476, help="Random seed for topology and training.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="GNN hidden dimension.")
    parser.add_argument("--gnn-layers", type=int, default=2, help="Number of message-passing layers.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--batch-size", type=int, default=16, help="Replay batch size.")
    parser.add_argument("--replay-capacity", type=int, default=1024, help="Replay buffer capacity.")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon for exploration.")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Final epsilon for exploration.")
    parser.add_argument("--target-update", type=int, default=10, help="Episodes between target-network syncs.")
    parser.add_argument("--verbose", action="store_true", help="Show environment generation logs.")
    return parser


def build_training_env_args(args: argparse.Namespace, episode: int) -> argparse.Namespace:
    env_args = build_env_parser().parse_args([])
    env_args.n_router = args.n_router
    env_args.n_data = args.n_data
    env_args.request_type = "multipartite"
    env_args.multipartite_targets = args.multipartite_targets
    env_args.multipartite_max_hop = args.multipartite_max_hop
    env_args.topology_init_seed = args.seed + episode
    env_args.summary_json = None
    env_args.save_plot = None
    env_args.show_plot = False
    env_args.verbose = args.verbose
    return env_args


def epsilon_for_episode(args: argparse.Namespace, episode: int) -> float:
    if args.episodes <= 1:
        return args.epsilon_end
    progress = episode / float(args.episodes - 1)
    return max(args.epsilon_end, args.epsilon_start + progress * (args.epsilon_end - args.epsilon_start))


def create_episode_env(env_args: argparse.Namespace):
    if env_args.verbose:
        env = create_quantum_env(env_args)
        env.reset()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            env = create_quantum_env(env_args)
            env.reset()
    return env


def main() -> int:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    replay = ReplayBuffer(args.replay_capacity)
    policy = None
    target_policy = None
    optimizer = None
    rewards = []
    skipped_episodes = 0
    simulator_backend = None

    for episode in range(args.episodes):
        env_args = build_training_env_args(args, episode)
        env = create_episode_env(env_args)
        delay_model = build_delay_model(env_args)
        reward_weights = build_reward_weights(env_args)
        fidelity_model = build_fidelity_model(env_args)
        max_hops = args.multipartite_max_hop if args.multipartite_max_hop > 0 else None
        simulator = create_ghz_simulator(
            args.ghz_simulator,
            shots=args.ghz_shots,
            seed=args.seed + episode,
            gate_noise=args.ghz_gate_noise,
            readout_error=args.ghz_readout_error,
        )
        simulator_backend = simulator.name

        try:
            request = generate_multipartite_request(
                env.network,
                target_count=args.multipartite_targets,
                request_id=episode,
                ttl=max_hops,
            )
        except ValueError:
            skipped_episodes += 1
            continue

        selection_env = CenterSelectionEnv(
            env.network,
            request,
            delay_model=delay_model,
            reward_weights=reward_weights,
            fidelity_model=fidelity_model,
            max_hops=max_hops,
            ghz_simulator=simulator,
        )
        observation = selection_env.reset()
        if not observation.candidate_centers:
            skipped_episodes += 1
            continue

        if policy is None:
            policy = CenterDqnPolicy(
                node_feature_dim=observation.node_features.shape[1],
                edge_feature_dim=observation.edge_features.shape[1],
                hidden_dim=args.hidden_dim,
                num_layers=args.gnn_layers,
            ).to(device)
            target_policy = copy.deepcopy(policy).to(device)
            optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

        epsilon = epsilon_for_episode(args, episode)
        selection = policy.act(
            observation,
            epsilon=epsilon,
            device=device,
            rng=env.network.agent_generator,
        )
        _, reward, _, _ = selection_env.step(selection.center)
        replay.add(Transition(observation=observation, action=selection.center, reward=reward))
        rewards.append(float(reward))

        if optimizer is not None and len(replay) > 0:
            policy.train()
            batch = replay.sample(args.batch_size)
            loss = compute_one_step_dqn_loss(policy, batch, device=device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (
            target_policy is not None
            and policy is not None
            and args.target_update > 0
            and (episode + 1) % args.target_update == 0
        ):
            target_policy.load_state_dict(policy.state_dict())

    if policy is None:
        raise RuntimeError("No valid training episode was generated; checkpoint was not created.")

    metadata = {
        "episodes": int(args.episodes),
        "trained_episodes": int(len(rewards)),
        "skipped_episodes": int(skipped_episodes),
        "average_reward": float(sum(rewards) / len(rewards)) if rewards else 0.0,
        "last_reward": float(rewards[-1]) if rewards else 0.0,
        "ghz_simulator_backend": simulator_backend,
        "ghz_gate_noise": float(args.ghz_gate_noise),
        "ghz_readout_error": float(args.ghz_readout_error),
        "node_feature_names": observation.node_feature_names,
        "edge_feature_names": observation.edge_feature_names,
        "device": str(device),
    }
    save_policy_checkpoint(args.checkpoint, policy, optimizer=optimizer, metadata=metadata)

    print(json.dumps({"checkpoint": str(args.checkpoint.resolve()), **metadata}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
