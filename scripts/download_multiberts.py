"""Download all MultiBERT models from HuggingFace to a local cache.

Final checkpoints: seeds 0-24  (google/multiberts-seed_{n})
Intermediate checkpoints: seeds 0-4, steps 20k-2000k
  (google/multiberts-seed_{n}-step_{step}k)

Run in the background:
    python download_multiberts.py &
or with nohup:
    nohup python download_multiberts.py > download_multiberts.log 2>&1 &
"""

import sys
from huggingface_hub import snapshot_download


FINAL_SEEDS = range(25)          # seeds 0-24
INTERMEDIATE_SEEDS = range(5)    # seeds 0-4
INTERMEDIATE_STEPS = [
    20, 40, 60, 80, 100, 120, 140, 160, 180, 200,
    300, 400, 500, 600, 700, 800, 900,
    1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
]

repos = []
for seed in FINAL_SEEDS:
    repos.append(f"google/multiberts-seed_{seed}")
for seed in INTERMEDIATE_SEEDS:
    for step in INTERMEDIATE_STEPS:
        repos.append(f"google/multiberts-seed_{seed}-step_{step}k")

total = len(repos)
print(f"Downloading {total} repos", flush=True)
print(f"  {len(list(FINAL_SEEDS))} final checkpoints (seeds 0-24)", flush=True)
print(f"  {len(list(INTERMEDIATE_SEEDS)) * len(INTERMEDIATE_STEPS)} intermediate checkpoints (seeds 0-4)", flush=True)

for i, repo_id in enumerate(repos, 1):
    try:
        snapshot_download(repo_id=repo_id)
        print(f"[{i}/{total}] OK  {repo_id}", flush=True)
    except Exception as e:
        print(f"[{i}/{total}] ERR {repo_id}: {e}", file=sys.stderr, flush=True)

print("Done.", flush=True)
