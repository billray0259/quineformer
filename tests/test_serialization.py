import pytest
import torch
from transformers import BertForMaskedLM, BertConfig

from quineformer.serialization import (
    serialize,
    deserialize,
    vector_component_labels,
    encoder_layer_row_bounds,
    deserialize_encoder_layer,
)


# ── Shared fixtures ────────────────────────────────────────────────

BERT_BASE_CONFIG = dict(
    vocab_size=1000,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=256,
    max_position_embeddings=32,
    type_vocab_size=2,
)


@pytest.fixture(scope="session")
def tiny_bert():
    """A small random BERT to keep tests fast."""
    config = BertConfig(**BERT_BASE_CONFIG)
    return BertForMaskedLM(config).eval()


@pytest.fixture(scope="session")
def serialized(tiny_bert):
    with torch.no_grad():
        return serialize(tiny_bert)


# ── Shape & accounting tests ───────────────────────────────────────

class TestSerializeShape:
    def test_output_is_2d(self, serialized):
        assert serialized.dim() == 2

    def test_width_is_d_model_plus_one(self, serialized):
        d = BERT_BASE_CONFIG["hidden_size"]
        assert serialized.shape[1] == d + 1

    def test_expected_vector_count(self, serialized):
        d = BERT_BASE_CONFIG["hidden_size"]
        d_ff = BERT_BASE_CONFIG["intermediate_size"]
        vocab = BERT_BASE_CONFIG["vocab_size"]
        max_pos = BERT_BASE_CONFIG["max_position_embeddings"]
        n_type = BERT_BASE_CONFIG["type_vocab_size"]
        n_layers = BERT_BASE_CONFIG["num_hidden_layers"]
        expected = vocab + max_pos + n_type + 2 + n_layers * (d * 4 + d_ff * 2 + 6)
        assert serialized.shape[0] == expected

    def test_total_parameter_count(self, tiny_bert, serialized):
        d = BERT_BASE_CONFIG["hidden_size"]
        d_ff = BERT_BASE_CONFIG["intermediate_size"]
        n_layers = BERT_BASE_CONFIG["num_hidden_layers"]
        bert_params = sum(
            p.numel() for n, p in tiny_bert.named_parameters()
            if n.startswith("bert.")
        )
        n_vectors = serialized.shape[0]
        d_model_values = n_vectors * d
        # Absorbed biases: Q/K/V each have d per layer, W₁ has d_ff per layer
        absorbed = n_layers * (3 * d + d_ff)
        assert d_model_values + absorbed == bert_params


# ── Round-trip tests ───────────────────────────────────────────────

class TestRoundTrip:
    def test_exact_roundtrip(self, tiny_bert, serialized):
        """serialize → deserialize → load_state_dict recovers all params."""
        params = deserialize(serialized, tiny_bert)
        config = BertConfig(**BERT_BASE_CONFIG)
        m2 = BertForMaskedLM(config).eval()
        m2.bert.load_state_dict(params)

        for (n1, p1), (n2, p2) in zip(
            tiny_bert.bert.named_parameters(),
            m2.bert.named_parameters(),
        ):
            assert torch.equal(p1, p2), f"Mismatch in {n1}"

    def test_roundtrip_output_matches(self, tiny_bert, serialized):
        """Model loaded from deserialized params produces same BERT outputs."""
        params = deserialize(serialized, tiny_bert)
        config = BertConfig(**BERT_BASE_CONFIG)
        m2 = BertForMaskedLM(config).eval()
        m2.bert.load_state_dict(params)

        ids = torch.randint(0, BERT_BASE_CONFIG["vocab_size"], (1, 8))
        with torch.no_grad():
            out1 = tiny_bert.bert(ids).last_hidden_state
            out2 = m2.bert(ids).last_hidden_state
        assert torch.equal(out1, out2)

    def test_state_dict_keys_match(self, tiny_bert, serialized):
        """Deserialized dict has exactly the keys of bert's state_dict."""
        params = deserialize(serialized, tiny_bert)
        expected = set(tiny_bert.bert.state_dict().keys())
        # Filter to only the keys we set (excludes position_ids buffer etc.)
        actual = set(params.keys())
        # All our keys must exist in the model
        assert actual.issubset(expected), f"Extra keys: {actual - expected}"
        # All model params (not buffers) must be in our dict
        param_keys = {n for n, _ in tiny_bert.bert.named_parameters()}
        assert param_keys == actual, f"Missing: {param_keys - actual}"


# ── Differentiability tests ────────────────────────────────────────

class TestDifferentiability:
    def test_serialize_grad_flows_to_params(self, tiny_bert):
        """Gradients flow from serialize output back to model parameters."""
        result = serialize(tiny_bert)
        loss = result.sum()
        loss.backward()

        grads_found = 0
        for n, p in tiny_bert.named_parameters():
            if n.startswith("bert.") and p.grad is not None:
                grads_found += 1
            if p.grad is not None:
                p.grad = None  # clean up
        assert grads_found > 0

    def test_deserialize_grad_flows_to_data(self, tiny_bert):
        """Gradients flow from deserialized params back to data tensor."""
        with torch.no_grad():
            data = serialize(tiny_bert)
        data = data.detach().requires_grad_(True)
        params = deserialize(data, tiny_bert)

        loss = sum(p.sum() for p in params.values())
        loss.backward()

        assert data.grad is not None
        assert (data.grad != 0).any()

    def test_full_pipeline_grad(self, tiny_bert):
        """Gradients flow through serialize → deserialize → forward."""
        data = serialize(tiny_bert)
        data2 = data.detach().requires_grad_(True)
        params = deserialize(data2, tiny_bert)

        config = BertConfig(**BERT_BASE_CONFIG)
        m2 = BertForMaskedLM(config).eval()
        # Use functional_call for differentiable forward
        ids = torch.randint(0, BERT_BASE_CONFIG["vocab_size"], (1, 8))
        prefixed = {f"bert.{k}": v for k, v in params.items()}
        out = torch.func.functional_call(m2, prefixed, args=(ids,))
        loss = out.logits.sum()
        loss.backward()

        assert data2.grad is not None
        assert (data2.grad != 0).any()

    def test_gradient_correctness(self):
        """Gradients through serialize→deserialize match direct backprop.

        Given a loss L(model(x)), compute dL/d(param) two ways:
          1) Direct: backprop through the model normally.
          2) Via serialization: serialize → deserialize → functional_call,
             then read gradients from the serialized matrix.

        The serialized matrix gradient, re-deserialized into param shapes,
        must exactly match the direct param gradients.
        """
        config = BertConfig(**BERT_BASE_CONFIG)
        model = BertForMaskedLM(config).eval()
        ids = torch.randint(0, BERT_BASE_CONFIG["vocab_size"], (1, 8))

        # ── Path 1: direct backprop ───────────────────────────────
        out1 = model(ids)
        loss1 = out1.logits.sum()
        loss1.backward()
        direct_grads = {}
        for n, p in model.bert.named_parameters():
            direct_grads[n] = p.grad.clone()
            p.grad = None  # clean up

        # ── Path 2: serialize → deserialize → functional_call ─────
        data = serialize(model).detach().requires_grad_(True)
        params = deserialize(data, config)
        prefixed = {f"bert.{k}": v for k, v in params.items()}
        out2 = torch.func.functional_call(model, prefixed, args=(ids,))
        loss2 = out2.logits.sum()
        loss2.backward()

        # Deserialize the data gradient to get per-param gradients
        grad_params = deserialize(data.grad, config)

        for name in direct_grads:
            assert torch.allclose(direct_grads[name], grad_params[name], atol=1e-6), \
                f"Gradient mismatch in {name}: max diff {(direct_grads[name] - grad_params[name]).abs().max():.2e}"


# ── Config extraction tests ────────────────────────────────────────

class TestConfigExtraction:
    def test_deserialize_accepts_model(self, tiny_bert, serialized):
        params = deserialize(serialized, tiny_bert)
        assert isinstance(params, dict)

    def test_deserialize_accepts_bert_model(self, tiny_bert, serialized):
        params = deserialize(serialized, tiny_bert.bert)
        assert isinstance(params, dict)

    def test_deserialize_accepts_config(self, serialized):
        config = BertConfig(**BERT_BASE_CONFIG)
        params = deserialize(serialized, config)
        assert isinstance(params, dict)

    def test_deserialize_rejects_bad_type(self, serialized):
        with pytest.raises(TypeError):
            deserialize(serialized, "not a model")

    def test_serialize_rejects_bad_type(self):
        with pytest.raises(TypeError):
            serialize("not a model")


# ── Component label tests ──────────────────────────────────────────

class TestVectorComponentLabels:
    def test_length_matches_serialized(self, tiny_bert, serialized):
        labels = vector_component_labels(tiny_bert)
        assert len(labels) == serialized.shape[0]

    def test_accepts_config(self):
        config = BertConfig(**BERT_BASE_CONFIG)
        labels = vector_component_labels(config)
        d = BERT_BASE_CONFIG["hidden_size"]
        d_ff = BERT_BASE_CONFIG["intermediate_size"]
        vocab = BERT_BASE_CONFIG["vocab_size"]
        max_pos = BERT_BASE_CONFIG["max_position_embeddings"]
        n_type = BERT_BASE_CONFIG["type_vocab_size"]
        n_layers = BERT_BASE_CONFIG["num_hidden_layers"]
        expected = vocab + max_pos + n_type + 2 + n_layers * (d * 4 + d_ff * 2 + 6)
        assert len(labels) == expected

    def test_known_label_set(self):
        config = BertConfig(**BERT_BASE_CONFIG)
        labels = vector_component_labels(config)
        expected_types = {
            'word_emb', 'pos_emb', 'type_emb', 'emb_ln_gamma', 'emb_ln_beta',
            'Q', 'K', 'V', 'O', 'b_O', 'attn_ln_gamma', 'attn_ln_beta',
            'mlp_up', 'mlp_down', 'b_2', 'mlp_ln_gamma', 'mlp_ln_beta',
        }
        assert set(labels) == expected_types

    def test_global_embeddings_first(self):
        config = BertConfig(**BERT_BASE_CONFIG)
        labels = vector_component_labels(config)
        vocab = BERT_BASE_CONFIG["vocab_size"]
        assert all(l == 'word_emb' for l in labels[:vocab])

    def test_per_layer_order(self):
        config = BertConfig(**BERT_BASE_CONFIG)
        labels = vector_component_labels(config)
        d = BERT_BASE_CONFIG["hidden_size"]
        d_ff = BERT_BASE_CONFIG["intermediate_size"]
        vocab = BERT_BASE_CONFIG["vocab_size"]
        max_pos = BERT_BASE_CONFIG["max_position_embeddings"]
        n_type = BERT_BASE_CONFIG["type_vocab_size"]
        # Start of first layer
        start = vocab + max_pos + n_type + 2
        expected_layer = (
            ['Q'] * d + ['K'] * d + ['V'] * d + ['O'] * d
            + ['b_O', 'attn_ln_gamma', 'attn_ln_beta']
            + ['mlp_up'] * d_ff + ['mlp_down'] * d_ff
            + ['b_2', 'mlp_ln_gamma', 'mlp_ln_beta']
        )
        assert labels[start:start + len(expected_layer)] == expected_layer

    def test_rejects_bad_type(self):
        with pytest.raises(TypeError):
            vector_component_labels("not a config")


class TestEncoderLayerHelpers:
    def test_encoder_layer_row_bounds_match_expected_width(self):
        config = BertConfig(**BERT_BASE_CONFIG)
        start, end = encoder_layer_row_bounds(config, 0)
        d = BERT_BASE_CONFIG["hidden_size"]
        d_ff = BERT_BASE_CONFIG["intermediate_size"]
        assert end - start == 4 * d + 2 * d_ff + 6

    def test_encoder_layer_row_bounds_reject_bad_index(self):
        config = BertConfig(**BERT_BASE_CONFIG)
        with pytest.raises(IndexError):
            encoder_layer_row_bounds(config, config.num_hidden_layers)

    def test_deserialize_encoder_layer_matches_full_deserialize(self, tiny_bert, serialized):
        full = deserialize(serialized, tiny_bert)

        for layer_idx in range(BERT_BASE_CONFIG["num_hidden_layers"]):
            start, end = encoder_layer_row_bounds(tiny_bert, layer_idx)
            layer_params = deserialize_encoder_layer(serialized[start:end], tiny_bert)
            prefix = f"encoder.layer.{layer_idx}."
            expected = {
                key[len(prefix):]: value
                for key, value in full.items()
                if key.startswith(prefix)
            }
            assert set(layer_params) == set(expected)
            for key, value in layer_params.items():
                assert torch.equal(value, expected[key]), f"Mismatch for layer {layer_idx} key {key}"

    def test_deserialize_encoder_layer_rejects_bad_shape(self):
        config = BertConfig(**BERT_BASE_CONFIG)
        bad = torch.zeros(5, config.hidden_size + 1)
        with pytest.raises(ValueError):
            deserialize_encoder_layer(bad, config)
