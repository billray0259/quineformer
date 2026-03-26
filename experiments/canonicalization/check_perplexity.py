import sys, torch, importlib.util
from pathlib import Path

# Insert bias_absorption so the bare name 'run_v1' resolves to it
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bias_absorption"))
from run_v1 import load_and_serialize_all, get_reference_batch, TEST_SEEDS

# Load canonicalization/run_v1.py under a distinct module name to avoid the
# name collision (it also does `from run_v1 import …` internally, which will
# now find the already-loaded bias_absorption module in sys.modules).
_spec = importlib.util.spec_from_file_location(
    "canon_run_v1", Path(__file__).resolve().parent / "run_v1.py"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["canon_run_v1"] = _mod
_spec.loader.exec_module(_mod)

load_frozen_projection = _mod.load_frozen_projection
evaluate_pair_perplexity = _mod.evaluate_pair_perplexity
evaluate_naive_interpolation = _mod.evaluate_naive_interpolation
DEVICE = _mod.DEVICE
RESULTS_DIR = _mod.RESULTS_DIR
from quineformer.canonicalization import CanonicalizationModule
from transformers import BertForMaskedLM, BertTokenizer
from itertools import combinations

print("Loading data and checkpoint...")
serialized, config = load_and_serialize_all()
projection = load_frozen_projection().to(DEVICE)
tokenizer = BertTokenizer.from_pretrained("google/multiberts-seed_0")
ref_batch = get_reference_batch(tokenizer, num_samples=8, max_length=256)
shell = BertForMaskedLM(config).eval().to(DEVICE)

canon = CanonicalizationModule(vocab_size=config.vocab_size, d_model=config.hidden_size)
ckpt = RESULTS_DIR / "canonicalization_module.pt"
canon.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
canon = canon.to(DEVICE).eval()
print(f"Loaded checkpoint: {ckpt}")

# Pre-encode
absorbed = {}
for seed in TEST_SEEDS:
    with torch.no_grad():
        absorbed[seed] = projection.encode(serialized[seed].to(DEVICE)).cpu()

val_pairs = list(combinations(TEST_SEEDS, 2))
print(f"\n{'Pair':<14} {'Canon PPL':>10} {'Ensemble PPL':>13} {'Ratio':>8} {'Naive PPL':>10}")
print("-" * 60)
ratios, naive_ppls, ensemble_ppls, interp_ppls = [], [], [], []
for seed_i, seed_j in val_pairs:
    r = evaluate_pair_perplexity(
        canon, projection, absorbed, serialized,
        seed_i, seed_j, 0.5, config, ref_batch, shell,
    )
    naive = evaluate_naive_interpolation(
        projection, serialized, seed_i, seed_j, 0.5, config, ref_batch,
    )
    print(f"({seed_i}, {seed_j})         {r['interp_ppl']:>10.2f} {r['ensemble_ppl']:>13.2f} {r['ratio']:>8.3f} {naive:>10.2f}")
    ratios.append(r['ratio'])
    naive_ppls.append(naive)
    ensemble_ppls.append(r['ensemble_ppl'])
    interp_ppls.append(r['interp_ppl'])

print("-" * 60)
print(f"{'Mean':<14} {sum(interp_ppls)/len(interp_ppls):>10.2f} {sum(ensemble_ppls)/len(ensemble_ppls):>13.2f} {sum(ratios)/len(ratios):>8.3f} {sum(naive_ppls)/len(naive_ppls):>10.2f}")