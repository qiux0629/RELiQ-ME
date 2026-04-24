#!/usr/bin/env python3
import argparse
import contextlib
import io
import json
import os
import sys
import warnings
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

from env.constants import EVAL_SEEDS
from env.entanglementenv import EntanglementEnv
from env.quantum_network import QuantumNetwork
from visualization import visualize_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a standalone RELiQ quantum network environment."
    )
    parser.add_argument("--n-router", type=int, default=100, help="Number of nodes.")
    parser.add_argument(
        "--n-router-connected",
        type=int,
        default=None,
        help="Number of connected nodes. Defaults to --n-router.",
    )
    parser.add_argument(
        "--n-data", type=int, default=1, help="Number of routing requests/agents."
    )
    parser.add_argument(
        "--n-quantum",
        type=int,
        default=20,
        help="Maximum number of quantum links per edge.",
    )
    parser.add_argument(
        "--random-topology",
        type=int,
        choices=[0, 1],
        default=1,
        help="Use random topology generation.",
    )
    parser.add_argument(
        "--force-fixed-topology",
        action="store_true",
        help="Force fixed topology even if --random-topology=1.",
    )
    parser.add_argument(
        "--num-topologies-train",
        type=int,
        default=1,
        help="Number of random topology seeds to pre-generate.",
    )
    parser.add_argument(
        "--topology-init-seed",
        type=int,
        default=476,
        help="Seed used for topology generation.",
    )
    parser.add_argument(
        "--train-topology-allow-eval-seed",
        action="store_true",
        help="Allow evaluation seeds during seed generation.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=3,
        help="Observation/action max degree.",
    )
    parser.add_argument(
        "--node-degree",
        type=int,
        default=3,
        help="Degree used when generating topologies.",
    )
    parser.add_argument("--ttl", type=int, default=10, help="Request TTL.")
    parser.add_argument(
        "--episode-steps", type=int, default=30, help="Training episode length."
    )
    parser.add_argument(
        "--eval-episode-steps", type=int, default=30, help="Eval episode length."
    )
    parser.add_argument(
        "--adaptive-episode-length",
        action="store_true",
        help="Enable adaptive episode length in the network.",
    )
    parser.add_argument(
        "--attenuation-coefficient",
        type=float,
        default=0.2,
        help="Fiber attenuation coefficient.",
    )
    parser.add_argument(
        "--link-decay",
        type=float,
        default=0.005,
        help="Per-step link decay used to derive timestep decay.",
    )
    parser.add_argument(
        "--swap-prob", type=float, default=1.0, help="Average swap probability."
    )
    parser.add_argument(
        "--swap-prob-std",
        type=float,
        default=0.0,
        help="Swap probability standard deviation.",
    )
    parser.add_argument(
        "--refresh-rate",
        type=float,
        default=0.0,
        help="Refresh rate for quantum links.",
    )
    parser.add_argument(
        "--fixed-path-length",
        type=int,
        default=-1,
        help="Required minimum path length for generated topologies.",
    )
    parser.add_argument(
        "--use-realistic-decay",
        action="store_true",
        help="Use realistic link decay model.",
    )
    parser.add_argument(
        "--decoupling-pulses-avg",
        type=int,
        default=1024,
        help="Average number of decoupling pulses.",
    )
    parser.add_argument(
        "--decoupling-pulses-std",
        type=int,
        default=0,
        help="Decoupling pulses standard deviation.",
    )
    parser.add_argument(
        "--initial-fidelity", type=float, default=1.0, help="Initial link fidelity."
    )
    parser.add_argument(
        "--auto-distillation-threshold",
        type=float,
        default=0.95,
        help="Automatic distillation threshold.",
    )
    parser.add_argument(
        "--message-speed",
        type=int,
        default=1,
        help="Coordination message speed.",
    )
    parser.add_argument(
        "--topohub-topology",
        type=str,
        default=None,
        help="Optional Topohub topology name.",
    )
    parser.add_argument(
        "--no-congestion",
        action="store_true",
        help="Disable congestion constraints.",
    )
    parser.add_argument(
        "--action-mask",
        dest="enable_action_mask",
        action="store_true",
        help="Enable action masking.",
    )
    parser.add_argument(
        "--infinite-quantum-links",
        action="store_true",
        help="Treat quantum links as infinite.",
    )
    parser.add_argument(
        "--no-idle-action",
        action="store_true",
        help="Disable idle action.",
    )
    parser.add_argument(
        "--fixed-request-rate",
        action="store_true",
        help="Use fixed request delay.",
    )
    parser.add_argument(
        "--fixed-requests",
        action="store_true",
        help="Keep request slots fixed across resets.",
    )
    parser.add_argument(
        "--request-based-observation",
        action="store_true",
        help="Use request-based node observations.",
    )
    parser.add_argument(
        "--reward-idle-punishment",
        type=float,
        default=0.75,
        help="Idle-action punishment weight.",
    )
    parser.add_argument(
        "--ignored-node-observations",
        type=int,
        default=0,
        help="Node-observation ignore bitmask.",
    )
    parser.add_argument(
        "--disable-agent-observation",
        action="store_true",
        help="Disable agent-specific observation fields.",
    )
    parser.add_argument(
        "--fairness", action="store_true", help="Enable fairness mode."
    )
    parser.add_argument(
        "--eval-max-entanglements",
        type=int,
        default=10,
        help="Maximum entanglements tracked in eval info.",
    )
    parser.add_argument(
        "--max-path-length",
        type=int,
        default=12,
        help="Maximum request path length.",
    )
    parser.add_argument(
        "--min-path-length",
        type=int,
        default=1,
        help="Minimum request path length.",
    )
    parser.add_argument(
        "--node-failure-frequency",
        type=int,
        default=-1,
        help="Node failure frequency.",
    )
    parser.add_argument(
        "--n-pseudo-targets",
        type=int,
        default=None,
        help="Number of pseudo targets.",
    )
    parser.add_argument(
        "--detailed-eval-logs",
        action="store_true",
        help="Enable detailed evaluation information.",
    )
    parser.add_argument(
        "--render",
        type=int,
        default=-1,
        help="Episode index to render. -1 disables in-environment rendering.",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        default=None,
        help="Save a visualization of the generated network environment to this path.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Show the generated network environment visualization.",
    )
    parser.add_argument(
        "--no-node-labels",
        action="store_true",
        help="Hide node ids in the saved environment visualization.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Write environment summary as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show internal environment generation logs.",
    )
    return parser


def create_quantum_env(args: argparse.Namespace) -> EntanglementEnv:
    network = QuantumNetwork(
        n_nodes=args.n_router,
        n_nodes_connected=args.n_router_connected,
        random_topology=args.random_topology if not args.force_fixed_topology else 0,
        n_random_seeds=args.num_topologies_train,
        topology_init_seed=args.topology_init_seed,
        excluded_seeds=None if args.train_topology_allow_eval_seed else EVAL_SEEDS,
        n_quantum_links=args.n_quantum,
        attenuation_coefficient=args.attenuation_coefficient,
        episode_steps=args.episode_steps,
        eval_episode_steps=args.eval_episode_steps,
        adaptive_episode_length=args.adaptive_episode_length,
        timestep_decay=1 - args.link_decay,
        swap_probability=args.swap_prob,
        swap_probability_std=args.swap_prob_std,
        neighbor_count=args.neighbors,
        node_degree=args.node_degree,
        ttl=args.ttl,
        topohub_topology=args.topohub_topology,
        refresh_rate=args.refresh_rate,
        fixed_path_length=args.fixed_path_length,
        use_realistic_decay=args.use_realistic_decay,
        n_decoupling_pulses_avg=args.decoupling_pulses_avg,
        n_decoupling_pulses_std=args.decoupling_pulses_std,
        world_width=1600 if args.use_realistic_decay else 2000,
        world_height=1600 if args.use_realistic_decay else 2000,
        initial_fidelity=args.initial_fidelity,
        auto_distillation_threshold=args.auto_distillation_threshold,
        message_speed=args.message_speed,
    )

    return EntanglementEnv(
        network,
        args.n_data,
        enable_congestion=not args.no_congestion,
        enable_action_mask=args.enable_action_mask,
        ttl=args.ttl,
        infinite_quantum_links=args.infinite_quantum_links,
        no_idle_action=args.no_idle_action,
        fixed_request_delay=args.fixed_request_rate,
        fixed_requests=args.fixed_requests,
        render=args.render,
        request_based_observation=args.request_based_observation,
        reward_idle_punishment=args.reward_idle_punishment,
        fidelity_choices=1,
        fixed_path_length=args.fixed_path_length,
        node_observation_ignore_value=args.ignored_node_observations,
        disable_agent_observation=args.disable_agent_observation,
        fairness=args.fairness,
        eval_max_entanglements=args.eval_max_entanglements,
        max_path_length=args.max_path_length,
        min_path_length=args.min_path_length,
        failure_frequency=args.node_failure_frequency,
        n_pseudo_targets=args.n_pseudo_targets,
        detailed_eval=args.detailed_eval_logs,
    )


def build_summary(env: EntanglementEnv, agent_obs, agent_adj):
    node_obs = env.get_node_observation()
    node_adj = env.get_nodes_adjacency()
    node_agent_matrix = env.get_node_agent_matrix()

    return {
        "topology_seed": int(env.network.current_topology_seed),
        "n_nodes": int(env.get_num_nodes()),
        "n_edges": int(len(env.network.edges)),
        "n_agents": int(env.get_num_agents()),
        "agent_observation_shape": [int(v) for v in agent_obs.shape],
        "agent_adjacency_shape": [int(v) for v in agent_adj.shape],
        "node_observation_shape": [int(v) for v in node_obs.shape],
        "node_adjacency_shape": [int(v) for v in node_adj.shape],
        "node_agent_matrix_shape": [int(v) for v in node_agent_matrix.shape],
        "diameter": int(env.network.get_diameter()),
        "requests": [
            {
                "id": int(request.id),
                "start": int(request.start),
                "target": int(request.target),
                "ttl": int(request.ttl),
            }
            for request in env.requests
        ],
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        if args.verbose:
            env = create_quantum_env(args)
            agent_obs, agent_adj = env.reset()
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                env = create_quantum_env(args)
                agent_obs, agent_adj = env.reset()

    summary = build_summary(env, agent_obs, agent_adj)

    if args.save_plot is not None:
        plot_path = args.save_plot.resolve()
        visualize_environment(
            env,
            filename=plot_path,
            show_plot=args.show_plot,
            title=f"Quantum Network Environment (seed={summary['topology_seed']})",
            label_nodes=not args.no_node_labels,
        )
        summary["topology_plot"] = str(plot_path)
    elif args.show_plot:
        visualize_environment(
            env,
            show_plot=True,
            title=f"Quantum Network Environment (seed={summary['topology_seed']})",
            label_nodes=not args.no_node_labels,
        )

    if args.summary_json is not None:
        json_path = args.summary_json.resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
