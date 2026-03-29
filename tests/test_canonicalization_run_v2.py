import importlib.util
from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertForMaskedLM

from quineformer.bias_absorption import BiasProjection
from quineformer.serialization import deserialize_encoder_layer, encoder_layer_row_bounds, serialize


RUN_V2_PATH = Path(__file__).resolve().parents[1] / "experiments" / "canonicalization" / "run_v2.py"
RUN_V2_SPEC = importlib.util.spec_from_file_location("canonicalization_run_v2", RUN_V2_PATH)
assert RUN_V2_SPEC is not None and RUN_V2_SPEC.loader is not None
run_v2 = importlib.util.module_from_spec(RUN_V2_SPEC)
RUN_V2_SPEC.loader.exec_module(run_v2)


def test_build_interpolated_layer_params_decodes_back_into_endpoint_basis():
    config = BertConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=8,
        type_vocab_size=2,
    )
    model = BertForMaskedLM(config).eval()
    projection = BiasProjection(config.hidden_size)
    projection.eval()

    serialized = serialize(model)
    start, end = encoder_layer_row_bounds(config, 0)
    absorbed = serialized[:, : config.hidden_size]
    permutation = torch.eye(config.hidden_size)
    permutation = permutation[torch.tensor([1, 0, 3, 2, 5, 4, 7, 6])]

    interpolated = run_v2.build_interpolated_layer_params(
        projection,
        absorbed,
        absorbed,
        permutation,
        permutation,
        permutation.transpose(0, 1),
        layer_idx=0,
        alpha=0.0,
        config=config,
        device=torch.device("cpu"),
    )

    restored_rows = run_v2.restore_layer_rows(
        projection,
        absorbed[start:end],
        config,
        layer_idx=0,
        device=torch.device("cpu"),
    )
    expected = deserialize_encoder_layer(restored_rows, config)

    for name, tensor in expected.items():
        assert torch.allclose(interpolated[name], tensor)


def test_safe_exp_saturates_on_overflow_input():
    assert run_v2.safe_exp(run_v2.MAX_EXP_INPUT + 1.0) == float("inf")


def test_linear_tau_schedule_hits_endpoints():
    assert run_v2.linear_tau_schedule(0, 5, 1.0, 0.05) == 1.0
    assert run_v2.linear_tau_schedule(4, 5, 1.0, 0.05) == pytest.approx(0.05)


def test_project_to_hard_permutation_returns_valid_assignment():
    soft = torch.tensor(
        [
            [0.9, 0.1, 0.0],
            [0.3, 0.4, 0.3],
            [0.0, 0.6, 0.4],
        ]
    )

    hard = run_v2.project_to_hard_permutation(soft)

    assert torch.equal(hard.sum(dim=0), torch.ones(3))
    assert torch.equal(hard.sum(dim=1), torch.ones(3))
    assert set(hard.unique().tolist()) <= {0.0, 1.0}