import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_states", default="/scratch/scholar/ngetach/hidden_states.pt",
                    help="Path to hidden_states.pt produced by collect_hidden_states.py")
parser.add_argument("--k", type=int, default=64,
                    help="Number of top principal components to retain")
parser.add_argument("--output", default="/scratch/scholar/ngetach/subspace.pt",
                    help="Path to save U and projection matrix P")
args = parser.parse_args()

# ── 1. Load hidden states ────────────────────────────────────────────────────
print(f"Loading hidden states from {args.hidden_states}...")
data = torch.load(args.hidden_states, weights_only=False)
H = data["hidden_states"].to(torch.float32)  # [n, d]
n, d = H.shape
print(f"H shape: {H.shape}  (n={n}, d={d})")

# ── 2. Center ────────────────────────────────────────────────────────────────
mean = H.mean(dim=0)           # [d]
H_centered = H - mean          # [n, d]

# ── 3. PCA via SVD ───────────────────────────────────────────────────────────
# torch.linalg.svd on the centered matrix: H_centered = U_svd @ S @ Vh
# The top-k right singular vectors (rows of Vh) are the principal directions.
k = min(args.k, n, d)
print(f"Running SVD to extract top k={k} principal components...")

# Use full_matrices=False for economy SVD
_, S_vals, Vh = torch.linalg.svd(H_centered, full_matrices=False)  # Vh: [min(n,d), d]

U = Vh[:k].T  # [d, k] — orthonormal basis of the forbidden subspace S

explained = (S_vals[:k] ** 2).sum() / (S_vals ** 2).sum()
print(f"Variance explained by top {k} components: {explained.item():.4f}")

# ── 4. Projection matrix P⊥ = I - U U^T ─────────────────────────────────────
# Stored explicitly; shape [d, d] = [4096, 4096].
# For the weight-modification step you will compute:  W_new = W @ P_perp
I = torch.eye(d, dtype=torch.float32)
P_perp = I - U @ U.T          # [d, d]

# ── 5. Save ──────────────────────────────────────────────────────────────────
torch.save({
    "U": U,               # [d, k]  orthonormal basis of S
    "P_perp": P_perp,     # [d, d]  projection onto complement of S
    "mean": mean,         # [d]     mean hidden state used for centering
    "k": k,
    "explained_variance": explained.item(),
    "singular_values": S_vals.cpu(),
}, args.output)

print(f"Saved U ({U.shape}), P_perp ({P_perp.shape}) to {args.output}")
