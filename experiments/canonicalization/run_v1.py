# %%
import torch
from transformers import BertTokenizer, BertForMaskedLM
from quineformer.canonicalization import CanonicalizationModule
import torch.nn.functional as F

# %%
HF_CACHE = "data/multiberts/hf_cache"
NUM_SEEDS = 25

embeddings = []
for seed in range(NUM_SEEDS):
    model_id = f"google/multiberts-seed_{seed}"
    m = BertForMaskedLM.from_pretrained(model_id, cache_dir=HF_CACHE)
    emb = m.bert.embeddings.word_embeddings.weight.detach().clone()
    embeddings.append(emb)
    del m
    print(f"Loaded seed {seed:2d}  shape={emb.shape}")

# Stack into dataset tensor: (25, 30522, 768)
E_all = torch.stack(embeddings)
print(f"\nDataset tensor shape: {E_all.shape}  dtype={E_all.dtype}")


# %%
from torch.utils.checkpoint import checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

E_gpu = E_all.to(DEVICE, dtype=torch.bfloat16)
N   = E_gpu.shape[0]   # 25
V   = E_gpu.shape[1]   # 30522
D   = E_gpu.shape[2]   # 768
VD  = V * D
print(f"E_gpu dtype={E_gpu.dtype}  size={E_gpu.numel()*E_gpu.element_size()/1024**3:.2f} GB")

canon_train = CanonicalizationModule(vocab_size=V, d_model=D).to(DEVICE)

# ── Hyperparameters ──────────────────────────────────────────────────────────
K            = 5      # permuted copies per seed
EPOCHS       = 500
LR           = 3e-4
ALPHA_DISCRIM = 1.0   # relative weight of discriminability vs consistency
LAMBDA_SHARP = 0.01
LOG_EVERY    = 50

optimizer = torch.optim.AdamW(canon_train.parameters(), lr=LR, weight_decay=1e-4)

# ── Pre-generate K fixed permutations per seed (CPU, just indices) ───────────
torch.manual_seed(42)
perms = torch.stack([
    torch.stack([torch.randperm(D) for _ in range(K)])
    for _ in range(N)
])  # (N, K, D)

# ── Pairwise MSE helpers ─────────────────────────────────────────────────────
pair_mask = torch.triu(torch.ones(N,   N,   dtype=torch.bool, device=DEVICE), diagonal=1)
triu_k    = torch.triu(torch.ones(K+1, K+1, dtype=torch.bool, device=DEVICE), diagonal=1)

def pairwise_mse(X, n):
    flat     = X.reshape(n, -1)
    norms_sq = (flat * flat).sum(dim=1)
    gram     = flat @ flat.T
    return (norms_sq[:, None] + norms_sq[None, :] - 2 * gram) / VD

# ── Margin from raw (uncanonicalized) embeddings ─────────────────────────────
with torch.inference_mode():
    raw_mean = pairwise_mse(E_gpu, N)[pair_mask].mean().item()
MARGIN = raw_mean * 0.5
print(f"Raw inter-seed MSE (no canon): {raw_mean:.6f}")
print(f"Margin set to:                 {MARGIN:.6f}")

def forward_fn(batch):
    return canon_train(batch)

# ── Training loop ─────────────────────────────────────────────────────────────
canon_train.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()

    # ── Term 1 — Consistency ──────────────────────────────────────────────────
    # Iterate one seed at a time; backward() after each seed to free the graph
    # so at most (K+1, V, D) ≈ 270 MB of activations are live at once.
    total_consist = 0.0
    for s in range(N):
        orig  = E_gpu[s]  # view, no allocation
        batch = torch.stack(
            [orig] + [orig[:, perms[s, k].to(DEVICE)] for k in range(K)]
        )  # (K+1, V, D)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            canon_b, _ = checkpoint(forward_fn, batch, use_reentrant=False)
        mse_s  = pairwise_mse(canon_b, K + 1)
        loss_s = mse_s[triu_k].mean() / N          # normalise so scale ≈ loss_discrim
        loss_s.backward()                           # accumulate grads, free graph
        total_consist += loss_s.detach().item()

    # ── Terms 2 & 3 — Discriminability + Sharpness ───────────────────────────
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        canonical_E, P = checkpoint(forward_fn, E_gpu, use_reentrant=False)
    mse_mat      = pairwise_mse(canonical_E, N)
    loss_discrim = F.relu(MARGIN - mse_mat[pair_mask]).mean()
    loss_sharp   = canon_train.row_entropy(P.float())
    loss_rest    = ALPHA_DISCRIM * loss_discrim + LAMBDA_SHARP * loss_sharp
    loss_rest.backward()

    torch.nn.utils.clip_grad_norm_(canon_train.parameters(), max_norm=1.0)
    optimizer.step()

    if epoch % LOG_EVERY == 0 or epoch == EPOCHS - 1:
        mean_cross  = mse_mat[pair_mask].mean().item()
        alloc_gb    = torch.cuda.memory_allocated(DEVICE) / 1024**3
        print(
            f"epoch {epoch:4d}  "
            f"consist {total_consist:.6f}  "
            f"discrim {loss_discrim.item():.6f}  "
            f"sharp {loss_sharp.item():.4f}  "
            f"cross_mse {mean_cross:.6f}  "
            f"tau {canon_train.tau.item():.4f}  "
            f"gpu {alloc_gb:.2f}GB"
        )



