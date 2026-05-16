from env.constants import EVAL_SEEDS
from env.entanglementenv import EntanglementEnv
from env.environment import EnvironmentVariant, NetworkEnv, reset_and_get_sizes
from env.quantum_network import (
    Edge,
    Entanglement,
    LinkReservation,
    QuantumLink,
    QuantumNetwork,
    QuantumRepeater,
)
from env.multipartite import PhysicalMemoryModel, StoredAtomicEntanglement

try:
    from env.gnn_encoder import (
        CenterSelectionHead,
        GNNEncoderOutput,
        GraphBatch,
        QuantumGraphGNNEncoder,
        encode_feature_package,
    )
    from env.center_policy import CenterPolicyOutput, MultipartiteCenterPolicy
    from env.routing_policy import BipartiteRoutingPolicy, RoutingPolicyOutput
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise

    _GNN_IMPORT_ERROR = exc

    def _missing_torch(*args, **kwargs):
        raise ModuleNotFoundError(
            "PyTorch is required for env.gnn_encoder. Install dependencies with "
            "`pip install -r standalone_quantum_env/requirements.txt`."
        ) from _GNN_IMPORT_ERROR

    CenterSelectionHead = _missing_torch
    BipartiteRoutingPolicy = _missing_torch
    CenterPolicyOutput = _missing_torch
    GNNEncoderOutput = _missing_torch
    GraphBatch = _missing_torch
    MultipartiteCenterPolicy = _missing_torch
    QuantumGraphGNNEncoder = _missing_torch
    RoutingPolicyOutput = _missing_torch
    encode_feature_package = _missing_torch

__all__ = [
    "BipartiteRoutingPolicy",
    "CenterSelectionHead",
    "CenterPolicyOutput",
    "EVAL_SEEDS",
    "Edge",
    "Entanglement",
    "EntanglementEnv",
    "EnvironmentVariant",
    "GNNEncoderOutput",
    "GraphBatch",
    "LinkReservation",
    "MultipartiteCenterPolicy",
    "NetworkEnv",
    "PhysicalMemoryModel",
    "QuantumGraphGNNEncoder",
    "QuantumLink",
    "QuantumNetwork",
    "QuantumRepeater",
    "RoutingPolicyOutput",
    "StoredAtomicEntanglement",
    "encode_feature_package",
    "reset_and_get_sizes",
]
