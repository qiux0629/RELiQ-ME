from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Dict, Optional, Sequence

import numpy as np


@dataclass
class GhzSimulationResult:
    backend: str
    requested_backend: str
    shots: int
    ghz_input_fidelity: float
    expected_ghz_fidelity: float
    ghz_fidelity: float
    success_probability: float
    measured_successes: int
    success: bool
    details: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "backend": self.backend,
            "requested_backend": self.requested_backend,
            "shots": int(self.shots),
            "ghz_input_fidelity": float(self.ghz_input_fidelity),
            "expected_ghz_fidelity": float(self.expected_ghz_fidelity),
            "ghz_fidelity": float(self.ghz_fidelity),
            "success_probability": float(self.success_probability),
            "measured_successes": int(self.measured_successes),
            "success": bool(self.success),
            "details": self.details,
        }


class GhzSimulatorBackend:
    name = "base"

    def __init__(
        self,
        shots: int = 1024,
        seed: Optional[int] = None,
        requested_backend: Optional[str] = None,
        gate_noise: float = 0.0,
        readout_error: float = 0.0,
    ):
        self.shots = int(shots)
        self.requested_backend = requested_backend or self.name
        self.rng = np.random.default_rng(seed)
        self.gate_noise = float(max(0.0, min(1.0, gate_noise)))
        self.readout_error = float(max(0.0, min(1.0, readout_error)))

    def simulate(
        self,
        path_fidelities: Sequence[float],
        fidelity_model,
    ) -> GhzSimulationResult:
        raise NotImplementedError

    @staticmethod
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

    def _model_probabilities(self, path_fidelities: Sequence[float], fidelity_model) -> tuple[float, float, float]:
        if not path_fidelities:
            return 0.0, 0.0, 0.0

        ghz_input_fidelity = float(np.prod(path_fidelities))
        expected_ghz_fidelity = self._apply_depolarizing_gate(
            ghz_input_fidelity,
            float(fidelity_model.ghz_fusion_gate_fidelity),
            len(path_fidelities),
        )
        bottleneck_fidelity = min(path_fidelities)
        success_probability = (
            float(fidelity_model.ghz_fusion_success_probability)
            if bottleneck_fidelity >= fidelity_model.threshold
            and expected_ghz_fidelity >= fidelity_model.threshold
            else 0.0
        )
        return ghz_input_fidelity, expected_ghz_fidelity, success_probability

    def _sample_result(
        self,
        path_fidelities: Sequence[float],
        fidelity_model,
        backend: str,
        details: Dict[str, object],
        ideal_success_ratio: float = 1.0,
    ) -> GhzSimulationResult:
        ghz_input_fidelity, expected_ghz_fidelity, success_probability = self._model_probabilities(
            path_fidelities,
            fidelity_model,
        )
        effective_probability = max(
            0.0,
            min(1.0, ideal_success_ratio * expected_ghz_fidelity * success_probability),
        )
        if self.shots > 0:
            measured_successes = int(self.rng.binomial(self.shots, effective_probability))
            ghz_fidelity = measured_successes / float(self.shots)
        else:
            measured_successes = int(effective_probability > 0)
            ghz_fidelity = effective_probability

        success = ghz_fidelity >= fidelity_model.threshold and success_probability > 0
        return GhzSimulationResult(
            backend=backend,
            requested_backend=self.requested_backend,
            shots=max(self.shots, 0),
            ghz_input_fidelity=ghz_input_fidelity,
            expected_ghz_fidelity=expected_ghz_fidelity,
            ghz_fidelity=float(ghz_fidelity if success else 0.0),
            success_probability=float(success_probability),
            measured_successes=measured_successes,
            success=bool(success),
            details=details,
        )

    def _exact_result(
        self,
        path_fidelities: Sequence[float],
        fidelity_model,
        backend: str,
        details: Dict[str, object],
        simulator_fidelity: float,
    ) -> GhzSimulationResult:
        ghz_input_fidelity, expected_ghz_fidelity, success_probability = self._model_probabilities(
            path_fidelities,
            fidelity_model,
        )
        exact_ghz_fidelity = max(
            0.0,
            min(1.0, ghz_input_fidelity * simulator_fidelity),
        )
        sampled_probability = exact_ghz_fidelity * success_probability
        if self.shots > 0:
            measured_successes = int(self.rng.binomial(self.shots, sampled_probability))
            sampled_ghz_fidelity = measured_successes / float(self.shots)
        else:
            measured_successes = int(sampled_probability > 0)
            sampled_ghz_fidelity = sampled_probability

        success = exact_ghz_fidelity >= fidelity_model.threshold and success_probability > 0
        details = {
            **details,
            "fidelity_source": "density_matrix_exact",
            "sampled_ghz_fidelity": float(sampled_ghz_fidelity),
        }
        return GhzSimulationResult(
            backend=backend,
            requested_backend=self.requested_backend,
            shots=max(self.shots, 0),
            ghz_input_fidelity=ghz_input_fidelity,
            expected_ghz_fidelity=expected_ghz_fidelity,
            ghz_fidelity=float(exact_ghz_fidelity if success else 0.0),
            success_probability=float(success_probability),
            measured_successes=measured_successes,
            success=bool(success),
            details=details,
        )


class NumpyGhzSimulator(GhzSimulatorBackend):
    name = "numpy"

    def simulate(
        self,
        path_fidelities: Sequence[float],
        fidelity_model,
    ) -> GhzSimulationResult:
        return self._sample_result(
            path_fidelities,
            fidelity_model,
            backend=self.name,
            details={"mode": "shot_sampling"},
        )


class QpandaGhzSimulator(GhzSimulatorBackend):
    name = "qpanda3"

    @staticmethod
    def is_available() -> bool:
        try:
            import_module("pyqpanda3.core")
            return True
        except Exception:
            return False

    def _build_ghz_program(self, n_qubits: int):
        core = import_module("pyqpanda3.core")
        circuit = core.QCircuit()
        circuit << core.H(0)
        for index in range(1, n_qubits):
            circuit << core.CNOT(index - 1, index)

        prog = core.QProg()
        prog << circuit
        return core, prog

    def _build_noise_model(self, core):
        noise_model = core.NoiseModel()
        if self.gate_noise > 0:
            gate_error = core.depolarizing_error(self.gate_noise)
            noise_model.add_all_qubit_quantum_error(
                gate_error,
                [core.GateType.H, core.GateType.CNOT],
            )
        if self.readout_error > 0:
            p = self.readout_error
            noise_model.add_all_qubit_read_out_error([[1.0 - p, p], [p, 1.0 - p]])
        return noise_model

    @staticmethod
    def _ghz_fidelity_from_density_matrix(density_matrix: np.ndarray, n_qubits: int) -> float:
        if n_qubits <= 0:
            return 0.0
        zero_index = 0
        one_index = (1 << n_qubits) - 1
        fidelity = 0.5 * (
            density_matrix[zero_index, zero_index]
            + density_matrix[one_index, one_index]
            + density_matrix[zero_index, one_index]
            + density_matrix[one_index, zero_index]
        )
        return float(np.clip(np.real(fidelity), 0.0, 1.0))

    def _run_noisy_density_simulation(self, n_qubits: int) -> tuple[float, Dict[str, object]]:
        try:
            core, prog = self._build_ghz_program(n_qubits)
            noise_model = self._build_noise_model(core)
            simulator = core.DensityMatrixSimulator()
            simulator.run(prog, noise_model)
            density_matrix = simulator.density_matrix()
            ideal_fidelity = self._ghz_fidelity_from_density_matrix(density_matrix, n_qubits)
            state_probabilities = {
                "0" * n_qubits: float(simulator.state_prob("0" * n_qubits)),
                "1" * n_qubits: float(simulator.state_prob("1" * n_qubits)),
            }
            return ideal_fidelity, {
                "density_matrix_shape": [int(v) for v in density_matrix.shape],
                "ideal_ghz_fidelity": ideal_fidelity,
                "basis_state_probabilities": state_probabilities,
            }
        except Exception as exc:
            raise RuntimeError(f"QPanda3 noisy GHZ simulation failed: {exc}") from exc

    def simulate(
        self,
        path_fidelities: Sequence[float],
        fidelity_model,
    ) -> GhzSimulationResult:
        if not path_fidelities:
            return self._sample_result(
                path_fidelities,
                fidelity_model,
                backend=self.name,
                details={"mode": "qpanda3_density_matrix", "noise_model": "depolarizing"},
            )

        ideal_ghz_fidelity, density_details = self._run_noisy_density_simulation(len(path_fidelities))
        return self._exact_result(
            path_fidelities,
            fidelity_model,
            backend=self.name,
            details={
                "mode": "qpanda3_density_matrix",
                "noise_model": "depolarizing",
                "gate_noise": self.gate_noise,
                "readout_error": self.readout_error,
                **density_details,
            },
            simulator_fidelity=ideal_ghz_fidelity,
        )


def create_ghz_simulator(
    backend: str = "auto",
    shots: int = 1024,
    seed: Optional[int] = None,
    gate_noise: float = 0.0,
    readout_error: float = 0.0,
) -> GhzSimulatorBackend:
    backend = backend.lower()
    if backend == "auto":
        if QpandaGhzSimulator.is_available():
            return QpandaGhzSimulator(
                shots=shots,
                seed=seed,
                requested_backend=backend,
                gate_noise=gate_noise,
                readout_error=readout_error,
            )
        return NumpyGhzSimulator(
            shots=shots,
            seed=seed,
            requested_backend=backend,
            gate_noise=gate_noise,
            readout_error=readout_error,
        )
    if backend == "numpy":
        return NumpyGhzSimulator(
            shots=shots,
            seed=seed,
            requested_backend=backend,
            gate_noise=gate_noise,
            readout_error=readout_error,
        )
    if backend == "qpanda":
        if not QpandaGhzSimulator.is_available():
            raise RuntimeError("QPanda3 is not installed. Install pyqpanda3 or use --ghz-simulator auto/numpy.")
        return QpandaGhzSimulator(
            shots=shots,
            seed=seed,
            requested_backend=backend,
            gate_noise=gate_noise,
            readout_error=readout_error,
        )
    raise ValueError(f"Unknown GHZ simulator backend: {backend}")
