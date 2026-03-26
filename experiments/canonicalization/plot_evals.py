"""Visualize canonicalization experiment V1 evaluation results.

Produces a multi-panel figure from whatever result files are available in
results_v1/:

  Panel 1 — Training loss curves (act + total + sharp) over epochs
  Panel 2 — Temperature τ and learning rate over epochs
  Panel 3 — Interpolation curves: PPL vs α for each validation pair
  Panel 4 — Per-pair PPL ratio at α=0.5 (canon-interp vs ensemble, naive)
  Panel 5 — Per-layer activation MSE for the sample pair

Panels are skipped gracefully if the corresponding result file doesn't yet
exist, so the script can be run mid-training.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results_v1"
OUTPUT_PNG = RESULTS_DIR / "evals.png"

COLORS = {
    "act": "#c44e52",
    "sharp": "#dd8452",
    "total": "#4c72b0",
    "tau": "#55a868",
    "lr": "#9467bd",
    "canon": "#4c72b0",
    "ensemble": "#222222",
    "naive": "#c44e52",
}

# ── Loaders ───────────────────────────────────────────────────────────────────


def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ── Individual panels ─────────────────────────────────────────────────────────


def plot_training_loss(ax: plt.Axes, log: list[dict]) -> None:
    epochs = [e["epoch"] for e in log]
    ax.plot(epochs, [e["loss_act"] for e in log],
            color=COLORS["act"], linewidth=2, label="Activation MSE")
    ax.plot(epochs, [e["loss_total"] for e in log],
            color=COLORS["total"], linewidth=2, linestyle="--", label="Total loss")
    ax.plot(epochs, [e["loss_sharp"] * 0.1 for e in log],
            color=COLORS["sharp"], linewidth=1.5, linestyle=":", label="λ·Entropy (λ=0.1)")
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)


def plot_tau_lr(ax: plt.Axes, log: list[dict]) -> None:
    epochs = [e["epoch"] for e in log]
    taus = [e["tau"] for e in log]
    lrs = [e["lr"] for e in log]

    ax2 = ax.twinx()
    ax.plot(epochs, taus, color=COLORS["tau"], linewidth=2, label="τ (temperature)")
    ax2.plot(epochs, lrs, color=COLORS["lr"], linewidth=1.5, linestyle="--", label="LR")

    ax.set_title("Temperature τ and Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("τ", color=COLORS["tau"])
    ax2.set_ylabel("Learning rate", color=COLORS["lr"])
    ax.tick_params(axis="y", colors=COLORS["tau"])
    ax2.tick_params(axis="y", colors=COLORS["lr"])
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)


def plot_interpolation_curves(ax: plt.Axes, curves: dict) -> None:
    cmap = plt.get_cmap("tab10")
    for idx, (pair_key, curve) in enumerate(curves.items()):
        alphas = [pt["alpha"] for pt in curve]
        interp_ppls = [pt["interp_ppl"] for pt in curve]
        ensemble_ppls = [pt["ensemble_ppl"] for pt in curve]

        color = cmap(idx)
        label_i = pair_key.replace("_", " vs seed ")
        ax.plot(alphas, interp_ppls, color=color, linewidth=2,
                marker="o", markersize=4, label=f"Canon-interp ({label_i})")
        ax.plot(alphas, ensemble_ppls, color=color, linewidth=1.5,
                linestyle="--", alpha=0.7, label=f"Ensemble ({label_i})")

    ax.set_title("Interpolation Curves: PPL vs α")
    ax.set_xlabel("Interpolation factor α")
    ax.set_ylabel("Perplexity")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def plot_ppl_ratios(ax: plt.Axes, results: dict) -> None:
    ppl = results.get("perplexity_results", {})
    naive = results.get("naive_interpolation", {})

    if not ppl:
        ax.text(0.5, 0.5, "No perplexity results yet", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("PPL Ratio at α=0.5")
        return

    pairs = sorted(ppl.keys())
    x = np.arange(len(pairs))
    width = 0.3

    canon_ratios = [ppl[k]["ratio"] for k in pairs]
    ensemble_ppls = [ppl[k]["ensemble_ppl"] for k in pairs]
    naive_ppls = [naive.get(k, float("nan")) for k in pairs]

    # Compute naive ratio relative to ensemble
    naive_ratios = [
        n / e if e > 0 else float("nan")
        for n, e in zip(naive_ppls, ensemble_ppls)
    ]

    ax.bar(x - width, canon_ratios, width, color=COLORS["canon"], alpha=0.85,
           label="Canon-interp / ensemble")
    ax.bar(x, naive_ratios, width, color=COLORS["naive"], alpha=0.85,
           label="Naive-interp / ensemble")
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1, label="Parity")

    mean_r = results.get("mean_perplexity_ratio", None)
    if mean_r is not None:
        ax.axhline(mean_r, color=COLORS["canon"], linestyle=":",
                   linewidth=1.5, label=f"Mean canon ratio = {mean_r:.3f}")

    labels = [k.replace("_", " vs ") for k in pairs]
    ax.set_xticks(x - width / 2)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_title("PPL Ratio at α=0.5 (vs Logit Ensemble)")
    ax.set_ylabel("PPL ratio (lower = better)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8)


def plot_per_layer_mse(ax: plt.Axes, secondary: dict) -> None:
    per_layer = secondary.get("per_layer_activation_mse")
    if not per_layer:
        ax.text(0.5, 0.5, "No per-layer MSE data yet", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Per-Layer Activation MSE")
        return

    layers = list(range(len(per_layer)))
    labels = ["Emb"] + [f"L{i}" for i in range(1, len(per_layer))]
    ax.bar(layers, per_layer, color=COLORS["canon"], alpha=0.8)
    ax.set_xticks(layers)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(
        f"Per-Layer Activation MSE "
        f"(pair {secondary.get('sample_pair', '?')})"
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3, axis="y")

    mean_mse = np.mean(per_layer)
    ax.axhline(mean_mse, color=COLORS["naive"], linestyle="--",
               linewidth=1.5, label=f"Mean = {mean_mse:.4f}")
    ax.legend(fontsize=9)


def plot_entropy_and_stability(ax: plt.Axes, secondary: dict) -> None:
    entropies = secondary.get("entropies", {})
    if not entropies:
        ax.text(0.5, 0.5, "No entropy data yet", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Permutation Sharpness (Row Entropy)")
        return

    seeds = sorted(entropies.keys(), key=int)
    values = [entropies[s] for s in seeds]
    x = np.arange(len(seeds))

    ax.bar(x, values, color=COLORS["tau"], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds], rotation=30, ha="right", fontsize=8)
    ax.set_title("P Row Entropy (↓ = sharper permutation)")
    ax.set_ylabel("Mean row entropy")
    ax.grid(True, alpha=0.3, axis="y")

    mean_ent = np.mean(values)
    ax.axhline(mean_ent, color="#444444", linestyle="--",
               linewidth=1, label=f"Mean = {mean_ent:.3f}")

    # Ideal: ln(768) for uniform, 0 for perfect permutation
    ax.axhline(0.0, color="#55a868", linestyle=":",
               linewidth=1, label="Perfect permutation (0)")
    ax.axhline(np.log(768), color=COLORS["naive"], linestyle=":",
               linewidth=1, label=f"Uniform (ln 768 = {np.log(768):.2f})")
    ax.legend(fontsize=8)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    training_log = load_json(RESULTS_DIR / "training_log.json")
    results = load_json(RESULTS_DIR / "results.json")
    curves = load_json(RESULTS_DIR / "interpolation_curves.json")
    secondary = results.get("secondary_metrics", {}) if results else {}

    # Decide which panels to draw
    has_log = bool(training_log)
    has_results = bool(results and results.get("perplexity_results"))
    has_curves = bool(curves)
    has_layer = bool(secondary.get("per_layer_activation_mse"))
    has_entropy = bool(secondary.get("entropies"))

    n_bottom = sum([has_curves, has_results, has_layer, has_entropy])
    n_rows = 2 if n_bottom else 1   # row 0: loss+tau   row 1: evals
    n_top_cols = 2
    n_bot_cols = max(n_bottom, 1)

    fig = plt.figure(figsize=(6 * n_bot_cols, 5 * n_rows), constrained_layout=True)
    fig.suptitle("Canonicalization Experiment V1", fontsize=14, fontweight="bold")

    gs_top = gridspec.GridSpec(1, n_top_cols, figure=fig,
                               top=1.0, bottom=0.5 + 0.02 * (n_rows == 1),
                               left=0.05, right=0.95, wspace=0.35)

    if has_log:
        ax_loss = fig.add_subplot(gs_top[0, 0])
        plot_training_loss(ax_loss, training_log)

        ax_tau = fig.add_subplot(gs_top[0, 1])
        plot_tau_lr(ax_tau, training_log)
    else:
        ax_msg = fig.add_subplot(gs_top[0, :])
        ax_msg.text(0.5, 0.5, "No training log yet", ha="center", va="center",
                    transform=ax_msg.transAxes, fontsize=14)

    if n_bottom > 0:
        gs_bot = gridspec.GridSpec(1, n_bot_cols, figure=fig,
                                   top=0.48, bottom=0.05,
                                   left=0.05, right=0.95, wspace=0.4)
        col = 0
        if has_curves:
            ax = fig.add_subplot(gs_bot[0, col]); col += 1
            plot_interpolation_curves(ax, curves)
        if has_results:
            ax = fig.add_subplot(gs_bot[0, col]); col += 1
            plot_ppl_ratios(ax, results)
        if has_layer:
            ax = fig.add_subplot(gs_bot[0, col]); col += 1
            plot_per_layer_mse(ax, secondary)
        if has_entropy:
            ax = fig.add_subplot(gs_bot[0, col]); col += 1
            plot_entropy_and_stability(ax, secondary)

    # Epoch label in title if partial run
    n_epochs_done = len(training_log) if training_log else 0
    if n_epochs_done > 0:
        last = training_log[-1]
        fig.suptitle(
            f"Canonicalization Experiment V1  "
            f"(epoch {n_epochs_done - 1}  "
            f"act={last['loss_act']:.4f}  "
            f"τ={last['tau']:.4f})",
            fontsize=13, fontweight="bold",
        )

    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUTPUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()
