import torch
import torch.nn as nn
import math


def sinkhorn(log_alpha: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """Sinkhorn normalization to produce a doubly-stochastic matrix from logits."""
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)


class CanonicalizationModule(nn.Module):
    """Learns to map embedding matrices to a canonical coordinate system.

    Architecture:
        1. W_down (vocab_size x d_model): projects E^T into coordinate representations
        2. Shallow transformer (2 layers, 4 heads): processes the d_model-length
           sequence of coordinate representations
        3. Sinkhorn normalization with learned temperature τ: converts transformer
           output into a soft permutation matrix P
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 768,
        n_layers: int = 2,
        n_heads: int = 4,
        sinkhorn_iters: int = 20,
        tau_init: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.sinkhorn_iters = sinkhorn_iters

        # Learnable projection from vocab space — produces coordinate representations
        self.W_down = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.normal_(self.W_down, std=1.0 / math.sqrt(vocab_size))

        # Shallow canonicalization transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Learned temperature for Sinkhorn sharpening
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def forward(
        self, E: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            E: Embedding matrices, shape (batch, vocab_size, d_model).

        Returns:
            canonical_E: Canonicalized embeddings, shape (batch, vocab_size, d_model).
            P: Soft permutation matrices, shape (batch, d_model, d_model).
        """
        # (batch, d_model, d_model) — each of the d_model coordinates gets a
        # d_model-dimensional representation via inner product with W_down
        S = torch.einsum("bvd,ve->bde", E, self.W_down)

        # Transformer processes the 768-length sequence of coordinate reps
        T = self.transformer(S)  # (batch, d_model, d_model)

        # Sinkhorn normalization → doubly-stochastic (soft permutation) matrix
        P = sinkhorn(T / self.tau, n_iters=self.sinkhorn_iters)

        # Apply P to reorder embedding columns into canonical form
        canonical_E = torch.bmm(E, P)  # (batch, vocab_size, d_model)

        return canonical_E, P

    def row_entropy(self, P: torch.Tensor) -> torch.Tensor:
        """Mean row entropy of P — used for sharpness regularization."""
        # P: (batch, d_model, d_model), each row sums to ~1
        ent = -(P * (P + 1e-12).log()).sum(dim=-1)  # (batch, d_model)
        return ent.mean()
