"""
For each k in {8, 32, 64, 128}, project the hidden states of three datasets
(toxicity, gsm8k, books) onto the first two principal components of the toxic
PCA subspace and save a scatter plot.

Each plot shows PC1 vs PC2 scores for a random subsample of vectors from each
dataset, colour-coded by dataset.
"""
import os
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRATCH = "/scratch/scholar/ngetach"

parser = argparse.ArgumentParser()
parser.add_argument("--toxic_states",  default=f"{SCRATCH}/hidden_states_toxic_7k.pt")
parser.add_argument("--gsm8k_states",  default=f"{SCRATCH}/hidden_states_gsm8k_7k.pt")
parser.add_argument("--books_states",  default=f"{SCRATCH}/hidden_states_books_7k.pt")
parser.add_argument("--subspace_dir",  default=SCRATCH)
parser.add_argument("--n_samples",     type=int, default=5000,
                    help="Vectors per dataset to plot (random subsample)")
parser.add_argument("--out_dir",       default="/home/ngetach/tml/plots")
parser.add_argument("--seed",          type=int, default=42)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
rng = np.random.default_rng(args.seed)

K_VALUES = [8, 32, 64, 128]
SUBSPACE_TMPL = os.path.join(args.subspace_dir, "subspace_masked_k{k}.pt")

DATASET_META = [
    ("Toxicity",  args.toxic_states,  "#d62728"),   # red
    ("GSM8K",     args.gsm8k_states,  "#2ca02c"),   # green
    ("Books",     args.books_states,  "#1f77b4"),   # blue
]

# ── load hidden states ────────────────────────────────────────────────────────
def load_states(path, n_samples, rng):
    data = torch.load(path, weights_only=False)
    H = data["hidden_states"].float()              # [N, 4096]
    if n_samples and H.shape[0] > n_samples:
        idx = rng.choice(H.shape[0], size=n_samples, replace=False)
        H = H[idx]
    return H

print("Loading hidden states...")
datasets = []
for label, path, color in DATASET_META:
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — skipping {label}")
        datasets.append((label, None, color))
        continue
    H = load_states(path, args.n_samples, rng)
    print(f"  {label}: {H.shape[0]:,} vectors")
    datasets.append((label, H, color))

# ── one plot per k ────────────────────────────────────────────────────────────
for k in K_VALUES:
    subspace_path = SUBSPACE_TMPL.format(k=k)
    if not os.path.exists(subspace_path):
        print(f"Subspace file not found: {subspace_path} — skipping k={k}")
        continue

    sub = torch.load(subspace_path, weights_only=False)
    U = sub["U"].float()                           # [4096, k]
    ev = sub["explained_variance"]
    pc1 = U[:, 0]                                  # [4096]
    pc2 = U[:, 1]                                  # [4096]

    fig, ax = plt.subplots(figsize=(7, 6))

    for label, H, color in datasets:
        if H is None:
            continue
        scores1 = (H @ pc1).numpy()                # [N]
        scores2 = (H @ pc2).numpy()                # [N]
        ax.scatter(scores1, scores2,
                   c=color, label=label,
                   alpha=0.25, s=4, linewidths=0)

    ax.set_xlabel("PC 1", fontsize=12)
    ax.set_ylabel("PC 2", fontsize=12)
    ax.set_title(
        f"PCA projection — k={k}  (expl. var = {ev:.1%})\n"
        f"Toxic subspace PC1 vs PC2",
        fontsize=12,
    )
    ax.legend(markerscale=4, framealpha=0.8)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    out_path = os.path.join(args.out_dir, f"pca_projection_k{k}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

print("Done.")
