import torch

from quineformer.canonicalization import CanonicalizationModule, sinkhorn


class TestCanonicalizationModule:
    def test_forward_returns_expected_shapes(self):
        model = CanonicalizationModule(
            vocab_size=13,
            d_model=8,
            sinkhorn_iters=10,
        )
        embeddings = torch.randn(4, 13, 8)

        canonical_embeddings, permutation = model(embeddings)

        assert canonical_embeddings.shape == (4, 13, 8)
        assert permutation.shape == (4, 8, 8)

    def test_forward_uses_qk_logits_for_p(self):
        model = CanonicalizationModule(
            vocab_size=13,
            d_model=8,
            sinkhorn_iters=10,
            tau_init=0.7,
        )
        embeddings = torch.randn(2, 13, 8)

        _, permutation = model(embeddings)

        embeddings_t = embeddings.permute(0, 2, 1)
        queries = embeddings_t @ model.W_q
        keys = embeddings_t @ model.W_k
        logits = (queries @ keys.permute(0, 2, 1)) / (model.d_model ** 0.5)
        expected = sinkhorn(logits / model.tau, n_iters=model.sinkhorn_iters)
        assert torch.allclose(permutation, expected, atol=1e-6)

    def test_sinkhorn_output_is_row_stochastic(self):
        model = CanonicalizationModule(
            vocab_size=13,
            d_model=8,
            sinkhorn_iters=20,
        )
        embeddings = torch.randn(3, 13, 8)

        _, permutation = model(embeddings)

        row_sums = permutation.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
        assert torch.all(permutation >= 0)