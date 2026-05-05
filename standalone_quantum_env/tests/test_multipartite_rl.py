from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from env.ghz_simulator import NumpyGhzSimulator, QpandaGhzSimulator, create_ghz_simulator
from env.multipartite import FidelityModel, calculate_center_selection_reward
from env.multipartite_rl import CenterDqnPolicy, CenterSelectionObservation


def test_center_selection_reward_uses_full_fidelity_on_success():
    reward = calculate_center_selection_reward(
        path_found=True,
        ghz_success=True,
        ghz_input_fidelity=0.8,
        ghz_fidelity=0.72,
    )
    assert reward == 0.72


def test_center_selection_reward_uses_half_input_fidelity_on_ghz_failure():
    reward = calculate_center_selection_reward(
        path_found=True,
        ghz_success=False,
        ghz_input_fidelity=0.8,
        ghz_fidelity=0.0,
    )
    assert reward == 0.4


def test_center_selection_reward_is_zero_without_path():
    reward = calculate_center_selection_reward(
        path_found=False,
        ghz_success=False,
        ghz_input_fidelity=0.8,
        ghz_fidelity=0.0,
    )
    assert reward == 0.0


def test_center_dqn_policy_masks_illegal_centers():
    observation = CenterSelectionObservation(
        node_features=np.zeros((4, 3), dtype=np.float32),
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
        edge_features=np.zeros((2, 2), dtype=np.float32),
        candidate_mask=np.array([False, True, False, True]),
        candidate_centers=[1, 3],
        terminals=[0, 2],
        node_feature_names=["a", "b", "is_center_candidate"],
        edge_feature_names=["edge_a", "edge_b"],
    )
    policy = CenterDqnPolicy(node_feature_dim=3, edge_feature_dim=2, hidden_dim=8, num_layers=1)
    q_values = policy(**observation.to_tensors())

    assert q_values[0] <= -1e8
    assert q_values[2] <= -1e8
    selection = policy.act(observation)
    assert selection.center in {1, 3}


def test_auto_ghz_simulator_falls_back_to_numpy_without_qpanda():
    simulator = create_ghz_simulator("auto", shots=16, seed=1)
    if QpandaGhzSimulator.is_available():
        assert simulator.name == "qpanda3"
    else:
        assert simulator.name == "numpy"


def test_numpy_ghz_simulator_returns_shot_result():
    simulator = NumpyGhzSimulator(shots=16, seed=1)
    result = simulator.simulate(
        [1.0, 1.0, 1.0],
        FidelityModel(
            threshold=0.1,
            ghz_fusion_gate_fidelity=1.0,
            ghz_fusion_success_probability=1.0,
        ),
    )

    assert result.backend == "numpy"
    assert result.shots == 16
    assert 0 <= result.measured_successes <= 16
    assert result.success


@pytest.mark.skipif(not QpandaGhzSimulator.is_available(), reason="pyqpanda3 is not installed")
def test_qpanda_ghz_simulator_uses_density_matrix_noise_model():
    simulator = create_ghz_simulator(
        "qpanda",
        shots=32,
        seed=1,
        gate_noise=0.05,
        readout_error=0.02,
    )
    result = simulator.simulate(
        [1.0, 1.0],
        FidelityModel(
            threshold=0.1,
            ghz_fusion_gate_fidelity=1.0,
            ghz_fusion_success_probability=1.0,
        ),
    )

    assert result.backend == "qpanda3"
    assert result.details["mode"] == "qpanda3_density_matrix"
    assert result.details["noise_model"] == "depolarizing"
    assert result.details["gate_noise"] == 0.05
    assert result.details["readout_error"] == 0.02
    assert result.details["density_matrix_shape"] == [4, 4]
    assert 0.0 <= result.details["ideal_ghz_fidelity"] <= 1.0
    assert result.details["fidelity_source"] == "density_matrix_exact"
    assert 0.0 <= result.details["sampled_ghz_fidelity"] <= 1.0
    assert result.ghz_fidelity <= result.ghz_input_fidelity
