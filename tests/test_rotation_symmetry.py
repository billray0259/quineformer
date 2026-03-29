import torch

from quineformer.rotation_symmetry import (
    interpolate_state_dicts,
    match_attention_qk_tensors,
    match_attention_vo_tensors,
    match_ffn_tensors,
    select_reference_seed,
)


def _permutation_matrix(indices: list[int]) -> torch.Tensor:
    perm = torch.zeros(len(indices), len(indices))
    for row, col in enumerate(indices):
        perm[row, col] = 1.0
    return perm


def _block_diag(*blocks: torch.Tensor) -> torch.Tensor:
    size = sum(block.shape[0] for block in blocks)
    out = torch.zeros(size, size)
    offset = 0
    for block in blocks:
        width = block.shape[0]
        out[offset : offset + width, offset : offset + width] = block
        offset += width
    return out


class TestMatchFfnTensors:
    def test_recovers_known_permutation(self):
        torch.manual_seed(0)
        anchor = {
            "W_I": torch.randn(6, 4),
            "B_I": torch.randn(6),
            "W_O_FFN": torch.randn(4, 6),
            "B_O_FFN": torch.randn(4),
        }
        permutation = _permutation_matrix([2, 5, 1, 4, 0, 3])
        local = {
            "W_I": permutation @ anchor["W_I"],
            "B_I": anchor["B_I"] @ permutation.transpose(0, 1),
            "W_O_FFN": anchor["W_O_FFN"] @ permutation.transpose(0, 1),
            "B_O_FFN": anchor["B_O_FFN"].clone(),
        }

        matched, metadata = match_ffn_tensors(local, anchor)

        assert torch.allclose(matched["W_I"], anchor["W_I"], atol=1e-5)
        assert torch.allclose(matched["B_I"], anchor["B_I"], atol=1e-5)
        assert torch.allclose(matched["W_O_FFN"], anchor["W_O_FFN"], atol=1e-5)
        assert torch.equal(metadata["permutation"].sum(dim=0), torch.ones(6))
        assert torch.equal(metadata["permutation"].sum(dim=1), torch.ones(6))


class TestMatchAttentionTensors:
    def test_qk_recovers_headwise_rotation(self):
        torch.manual_seed(0)
        anchor = {
            "W_Q": torch.randn(4, 3),
            "W_K": torch.randn(4, 3),
            "B_Q": torch.randn(4),
            "B_K": torch.randn(4),
        }
        rot_0 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        rot_1 = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        rotation = _block_diag(rot_0, rot_1)
        local = {
            "W_Q": rotation @ anchor["W_Q"],
            "W_K": rotation @ anchor["W_K"],
            "B_Q": anchor["B_Q"] @ rotation.transpose(0, 1),
            "B_K": anchor["B_K"] @ rotation.transpose(0, 1),
        }

        matched, metadata = match_attention_qk_tensors(
            local,
            anchor,
            num_attention_heads=2,
            attention_head_size=2,
        )

        assert torch.allclose(matched["W_Q"], anchor["W_Q"], atol=1e-5)
        assert torch.allclose(matched["W_K"], anchor["W_K"], atol=1e-5)
        assert torch.allclose(matched["B_Q"], anchor["B_Q"], atol=1e-5)
        assert torch.allclose(matched["B_K"], anchor["B_K"], atol=1e-5)
        assert len(metadata["rotations"]) == 2

    def test_vo_recovers_headwise_rotation(self):
        torch.manual_seed(0)
        anchor = {
            "W_V": torch.randn(4, 3),
            "W_O": torch.randn(3, 4),
            "B_V": torch.randn(4),
            "B_O": torch.randn(3),
        }
        rot_0 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        rot_1 = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        rotation = _block_diag(rot_0, rot_1)
        local = {
            "W_V": rotation @ anchor["W_V"],
            "W_O": anchor["W_O"] @ rotation.transpose(0, 1),
            "B_V": anchor["B_V"] @ rotation.transpose(0, 1),
            "B_O": anchor["B_O"].clone(),
        }

        matched, metadata = match_attention_vo_tensors(
            local,
            anchor,
            num_attention_heads=2,
            attention_head_size=2,
        )

        assert torch.allclose(matched["W_V"], anchor["W_V"], atol=1e-5)
        assert torch.allclose(matched["W_O"], anchor["W_O"], atol=1e-5)
        assert torch.allclose(matched["B_V"], anchor["B_V"], atol=1e-5)
        assert len(metadata["rotations"]) == 2


class TestReferenceSelection:
    def test_select_reference_seed_returns_medoid(self):
        serialized = {
            0: torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            1: torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
            2: torch.tensor([[3.0, 3.0], [3.0, 3.0]]),
        }

        reference = select_reference_seed(
            serialized,
            [0, 1, 2],
            vocab_size=2,
            hidden_size=2,
            metric="full",
        )

        assert reference == 1


class TestInterpolateStateDicts:
    def test_interpolates_only_floating_tensors(self):
        state_a = {
            "weight": torch.tensor([0.0, 2.0]),
            "buffer": torch.tensor([1, 2], dtype=torch.long),
        }
        state_b = {
            "weight": torch.tensor([2.0, 4.0]),
            "buffer": torch.tensor([9, 9], dtype=torch.long),
        }

        interpolated = interpolate_state_dicts(state_a, state_b, 0.5)

        assert torch.allclose(interpolated["weight"], torch.tensor([1.0, 3.0]))
        assert torch.equal(interpolated["buffer"], state_a["buffer"])