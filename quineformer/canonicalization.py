import torch
import torch.nn as nn
import math


def sinkhorn(log_alpha: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """Normalize logits with a row-wise softmax.

    The `n_iters` argument is preserved for API compatibility with existing
    experiment code, but is ignored.
    """
    del n_iters
    return torch.softmax(log_alpha, dim=-1)


class CanonicalizationModule(nn.Module):
    """Learns to map embedding matrices to a canonical coordinate system.

    Architecture:
        1. W_q (vocab_size x d_model): projects E^T into coordinate queries
        2. W_k (vocab_size x d_model): projects E^T into coordinate keys
          3. QK^T attention logits over coordinates are normalized row-wise to
              produce a soft transport matrix P
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 768,
        sinkhorn_iters: int = 20,
        tau_init: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.sinkhorn_iters = sinkhorn_iters

        self.W_q = nn.Parameter(torch.empty(vocab_size, d_model))
        self.W_k = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.normal_(self.W_q, std=1.0 / math.sqrt(vocab_size))
        nn.init.normal_(self.W_k, std=1.0 / math.sqrt(vocab_size))

        # Learned temperature for logit sharpening
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
        E_t = E.permute(0, 2, 1)
        Q = E_t @ self.W_q  # (batch, d_model, d_model)
        K = E_t @ self.W_k  # (batch, d_model, d_model)
        attention_logits = (Q @ K.permute(0, 2, 1)) / math.sqrt(self.d_model)

        # Row-wise normalization → stochastic transport matrix
        P = sinkhorn(attention_logits / self.tau, n_iters=self.sinkhorn_iters)

        # Apply P to reorder embedding columns into canonical form
        canonical_E = E @ P  # (batch, vocab_size, d_model)

        return canonical_E, P

    def row_entropy(self, P: torch.Tensor) -> torch.Tensor:
        """Mean row entropy of P — used for sharpness regularization."""
        # P: (batch, d_model, d_model), each row sums to ~1
        ent = -(P * (P + 1e-12).log()).sum(dim=-1)  # (batch, d_model)
        return ent.mean()
