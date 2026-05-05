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
from env.ghz_simulator import create_ghz_simulator
from env.multipartite import (
    DelayModel,
    FidelityModel,
    RewardWeights,
    build_gnn_feature_package,
    execute_multipartite_request,
    generate_multipartite_request,
    plan_multipartite_entanglement,
    plan_multipartite_entanglement_for_center,
)
from env.quantum_network import QuantumLink, QuantumNetwork
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
        "--reward-mode",
        choices=["legacy", "time_fidelity"],
        default="legacy",
        help="Reward function used by EntanglementEnv.step.",
    )
    parser.add_argument(
        "--reward-success-bonus",
        type=float,
        default=2.0,
        help="Bonus for successful end-to-end entanglement in time_fidelity mode.",
    )
    parser.add_argument(
        "--reward-path-found-bonus",
        type=float,
        default=0.5,
        help="Bonus for reaching the target in time_fidelity mode.",
    )
    parser.add_argument(
        "--reward-fidelity-weight",
        type=float,
        default=1.0,
        help="Fidelity reward weight in time_fidelity mode.",
    )
    parser.add_argument(
        "--reward-latency-weight",
        type=float,
        default=0.0,
        help="Elapsed-step penalty weight in time_fidelity mode.",
    )
    parser.add_argument(
        "--reward-resource-weight",
        type=float,
        default=0.0,
        help="Quantum-resource penalty weight in time_fidelity mode.",
    )
    parser.add_argument(
        "--request-type",
        choices=["bipartite", "multipartite"],
        default="bipartite",
        help="Request model to summarize. bipartite keeps the original RELiQ request model; multipartite generates source-to-many-target requests.",
    )
    parser.add_argument(
        "--multipartite-targets",
        type=int,
        default=3,
        help="Number of targets in a one-to-many multipartite request.",
    )
    parser.add_argument(
        "--multipartite-max-hop",
        type=int,
        default=0,
        help="Maximum allowed hops per terminal-to-center route. 0 disables this check.",
    )
    parser.add_argument(
        "--fidelity-threshold",
        type=float,
        default=QuantumLink.FIDELITY_THRESHOLD,
        help="Minimum usable route/GHZ fidelity.",
    )
    parser.add_argument(
        "--ghz-fusion-gate-fidelity",
        type=float,
        default=1.0,
        help="Gate fidelity applied to the final GHZ fusion depolarization model.",
    )
    parser.add_argument(
        "--ghz-fusion-success-probability",
        type=float,
        default=1.0,
        help="Estimated probability that the final GHZ fusion operation succeeds.",
    )
    parser.add_argument(
        "--multipartite-terminals",
        type=int,
        default=0,
        help="Legacy planner mode: build a GHZ-style plan by collecting this many endpoints from bipartite requests. 0 disables it.",
    )
    parser.add_argument(
        "--center-strategy",
        choices=["balanced", "median", "minimax", "min-latency", "max-fidelity", "random", "rl"],
        default="balanced",
        help="Center-node selection strategy for multipartite planning.",
    )
    parser.add_argument(
        "--rl-policy-path",
        type=Path,
        default=None,
        help="Checkpoint path for --center-strategy rl.",
    )
    parser.add_argument(
        "--ghz-simulator",
        choices=["auto", "qpanda", "numpy"],
        default="auto",
        help="GHZ simulation backend used by multipartite execution.",
    )
    parser.add_argument(
        "--ghz-shots",
        type=int,
        default=1024,
        help="Shots used by the GHZ simulation backend.",
    )
    parser.add_argument(
        "--ghz-gate-noise",
        type=float,
        default=0.0,
        help="Depolarizing noise probability applied to QPanda3 H/CNOT gates.",
    )
    parser.add_argument(
        "--ghz-readout-error",
        type=float,
        default=0.0,
        help="Readout bit-flip error probability applied by the QPanda3 noise model.",
    )
    parser.add_argument(
        "--delay-sample-time",
        type=float,
        default=1.0,
        help="Sampling delay term used by multipartite planning.",
    )
    parser.add_argument(
        "--delay-packet-time",
        type=float,
        default=1.0,
        help="Packetization delay term used by multipartite planning.",
    )
    parser.add_argument(
        "--delay-processing-time",
        type=float,
        default=1.0,
        help="Classical processing delay term used by multipartite planning.",
    )
    parser.add_argument(
        "--delay-agent-decision-time",
        type=float,
        default=1.0,
        help="Per-hop agent decision delay term used by multipartite planning.",
    )
    parser.add_argument(
        "--delay-gnn-inference-time",
        type=float,
        default=1.0,
        help="GNN inference delay term used by multipartite planning.",
    )
    parser.add_argument(
        "--delay-bsm-time",
        type=float,
        default=1.0,
        help="Bell-state measurement delay term used by multipartite planning.",
    )
    parser.add_argument(
        "--delay-memory-time",
        type=float,
        default=1.0,
        help="Memory-access delay term used by multipartite planning.",
    )
    parser.add_argument(
        "--delay-ghz-fusion-time",
        type=float,
        default=1.0,
        help="GHZ fusion delay term used by multipartite planning.",
    )
    parser.add_argument(
        "--delay-classical-scale",
        type=float,
        default=1.0,
        help="Multiplier for propagation delay in multipartite planning.",
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
        reward_mode=args.reward_mode,
        reward_success_bonus=args.reward_success_bonus,
        reward_path_found_bonus=args.reward_path_found_bonus,
        reward_fidelity_weight=args.reward_fidelity_weight,
        reward_latency_weight=args.reward_latency_weight,
        reward_resource_weight=args.reward_resource_weight,
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


def select_multipartite_terminals(env: EntanglementEnv, terminal_count: int) -> list[int]:
    terminals: list[int] = []
    for request in env.requests:
        for node_id in (request.start, request.target):
            if node_id not in terminals:
                terminals.append(int(node_id))
            if len(terminals) >= terminal_count:
                return terminals

    valid_nodes = [
        int(node.id)
        for node in env.network.nodes[: env.network.n_nodes_connected]
        if node.source_destination_valid and not node.virtual and node.id not in terminals
    ]
    env.network.agent_generator.shuffle(valid_nodes)
    terminals.extend(valid_nodes[: max(terminal_count - len(terminals), 0)])
    return terminals


def build_delay_model(args: argparse.Namespace) -> DelayModel:
    return DelayModel(
        sample_time=args.delay_sample_time,
        packet_time=args.delay_packet_time,
        processing_time=args.delay_processing_time,
        agent_decision_time=args.delay_agent_decision_time,
        gnn_inference_time=args.delay_gnn_inference_time,
        bsm_time=args.delay_bsm_time,
        memory_time=args.delay_memory_time,
        ghz_fusion_time=args.delay_ghz_fusion_time,
        classical_message_scale=args.delay_classical_scale,
    )


def build_reward_weights(args: argparse.Namespace) -> RewardWeights:
    return RewardWeights(
        success_bonus=args.reward_success_bonus,
        path_found_bonus=args.reward_path_found_bonus,
        fidelity_weight=args.reward_fidelity_weight,
        latency_weight=args.reward_latency_weight,
        resource_weight=args.reward_resource_weight,
    )


def build_fidelity_model(args: argparse.Namespace) -> FidelityModel:
    return FidelityModel(
        threshold=args.fidelity_threshold,
        ghz_fusion_gate_fidelity=args.ghz_fusion_gate_fidelity,
        ghz_fusion_success_probability=args.ghz_fusion_success_probability,
    )


def build_execution_simulator(args: argparse.Namespace, env: EntanglementEnv):
    return create_ghz_simulator(
        args.ghz_simulator,
        shots=args.ghz_shots,
        seed=int(env.network.current_topology_seed),
        gate_noise=args.ghz_gate_noise,
        readout_error=args.ghz_readout_error,
    )


def select_rl_center(args: argparse.Namespace, network, terminals: list[int]):
    if args.rl_policy_path is None:
        raise ValueError("--rl-policy-path is required when --center-strategy rl.")
    if not args.rl_policy_path.exists():
        raise FileNotFoundError(f"RL policy checkpoint not found: {args.rl_policy_path}")

    from env.multipartite_rl import load_policy_checkpoint, select_center_with_policy

    policy, metadata = load_policy_checkpoint(args.rl_policy_path)
    selection = select_center_with_policy(policy, network, terminals)
    return selection, metadata


def add_rl_summary_fields(summary: dict, args: argparse.Namespace, selection, metadata: dict, execution=None) -> None:
    summary["rl_policy"] = {
        "path": str(args.rl_policy_path.resolve()) if args.rl_policy_path is not None else None,
        "metadata": metadata,
    }
    summary["selected_center_q_value"] = float(selection.q_value)
    summary["candidate_q_values"] = {
        str(center): float(value)
        for center, value in selection.candidate_q_values.items()
    }
    if execution is not None and execution.rl_reward is not None:
        summary["rl_reward"] = float(execution.rl_reward)


def add_multipartite_summary(args: argparse.Namespace, env: EntanglementEnv, summary: dict) -> None:
    delay_model = build_delay_model(args)
    reward_weights = build_reward_weights(args)
    fidelity_model = build_fidelity_model(args)

    if args.request_type == "multipartite":
        max_hops = args.multipartite_max_hop if args.multipartite_max_hop > 0 else None
        try:
            ghz_simulator = build_execution_simulator(args, env)
            request = generate_multipartite_request(
                env.network,
                target_count=args.multipartite_targets,
                request_id=0,
                ttl=max_hops,
            )
            rl_selection = None
            rl_metadata = {}
            if args.center_strategy == "rl":
                rl_selection, rl_metadata = select_rl_center(args, env.network, request.terminals)
                execution = execute_multipartite_request(
                    env.network,
                    request,
                    center_strategy="rl",
                    delay_model=delay_model,
                    reward_weights=reward_weights,
                    fidelity_model=fidelity_model,
                    max_hops=max_hops,
                    center=rl_selection.center,
                    ghz_simulator=ghz_simulator,
                    use_rl_reward=True,
                    selected_center_q_value=rl_selection.q_value,
                )
            else:
                execution = execute_multipartite_request(
                    env.network,
                    request,
                    center_strategy=args.center_strategy,
                    delay_model=delay_model,
                    reward_weights=reward_weights,
                    fidelity_model=fidelity_model,
                    max_hops=max_hops,
                    ghz_simulator=ghz_simulator,
                )
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            summary["multipartite_execution_error"] = str(exc)
            return

        legacy_requests = summary.pop("requests")
        summary["request_type"] = "multipartite"
        summary["legacy_bipartite_requests"] = legacy_requests
        summary["requests"] = [request.to_dict()]
        summary["multipartite_execution"] = execution.to_dict()
        summary["multipartite_delay_model"] = delay_model.__dict__
        summary["multipartite_reward_weights"] = reward_weights.__dict__
        summary["multipartite_fidelity_model"] = fidelity_model.__dict__
        summary["ghz_simulator_backend"] = (
            execution.ghz_simulation.backend
            if execution.ghz_simulation is not None
            else ghz_simulator.name
        )
        if args.center_strategy == "rl" and rl_selection is not None:
            add_rl_summary_fields(summary, args, rl_selection, rl_metadata, execution=execution)
        gnn_features = build_gnn_feature_package(
            env.network,
            terminals=request.terminals,
            center=execution.plan.center,
        )
        summary["gnn_feature_package"] = {
            "node_features_shape": [int(v) for v in gnn_features["node_features"].shape],
            "edge_index_shape": [int(v) for v in gnn_features["edge_index"].shape],
            "edge_features_shape": [int(v) for v in gnn_features["edge_features"].shape],
            "node_feature_names": gnn_features["node_feature_names"],
            "edge_feature_names": gnn_features["edge_feature_names"],
        }
        return

    summary["request_type"] = "bipartite"
    if args.multipartite_terminals <= 0:
        return

    terminals = select_multipartite_terminals(env, args.multipartite_terminals)
    if len(terminals) < 2:
        summary["multipartite_plan_error"] = "At least two valid terminals are required."
        return

    try:
        if args.center_strategy == "rl":
            rl_selection, rl_metadata = select_rl_center(args, env.network, terminals)
            plan = plan_multipartite_entanglement_for_center(
                env.network,
                terminals,
                center=rl_selection.center,
                center_strategy="rl",
                delay_model=delay_model,
                reward_weights=reward_weights,
                fidelity_model=fidelity_model,
            )
            add_rl_summary_fields(summary, args, rl_selection, rl_metadata)
        else:
            plan = plan_multipartite_entanglement(
                env.network,
                terminals,
                center_strategy=args.center_strategy,
                delay_model=delay_model,
                reward_weights=reward_weights,
                fidelity_model=fidelity_model,
            )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        summary["multipartite_plan_error"] = str(exc)
        return

    gnn_features = build_gnn_feature_package(env.network, terminals=terminals, center=plan.center)
    summary["multipartite_plan"] = plan.to_dict()
    summary["multipartite_delay_model"] = delay_model.__dict__
    summary["multipartite_reward_weights"] = reward_weights.__dict__
    summary["multipartite_fidelity_model"] = fidelity_model.__dict__
    summary["gnn_feature_package"] = {
        "node_features_shape": [int(v) for v in gnn_features["node_features"].shape],
        "edge_index_shape": [int(v) for v in gnn_features["edge_index"].shape],
        "edge_features_shape": [int(v) for v in gnn_features["edge_features"].shape],
        "node_feature_names": gnn_features["node_feature_names"],
        "edge_feature_names": gnn_features["edge_feature_names"],
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
    add_multipartite_summary(args, env, summary)

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
