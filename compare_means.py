"""
Computes the mean hidden state vector for the toxic and GSM8K datasets,
then reports the difference vector and its norm.
"""
import torch

TOXIC_PATH = "/scratch/scholar/ngetach/hidden_states_masked.pt"
GSM8K_PATH = "/scratch/scholar/ngetach/hidden_states_gsm8k.pt"

toxic = torch.load(TOXIC_PATH, weights_only=False)
gsm8k = torch.load(GSM8K_PATH, weights_only=False)

toxic_hs = toxic["hidden_states"].float()   # [N1, 4096]
gsm8k_hs = gsm8k["hidden_states"].float()   # [N2, 4096]

toxic_mean = toxic_hs.mean(dim=0)           # [4096]
gsm8k_mean = gsm8k_hs.mean(dim=0)          # [4096]
diff = toxic_mean - gsm8k_mean              # [4096]

print(f"Toxic hidden states : {toxic_hs.shape}")
print(f"GSM8K hidden states : {gsm8k_hs.shape}")
print()
print(f"Toxic mean norm     : {toxic_mean.norm().item():.4f}")
print(f"GSM8K mean norm     : {gsm8k_mean.norm().item():.4f}")
print(f"Difference norm     : {diff.norm().item():.4f}")
print(f"Cosine similarity   : {torch.nn.functional.cosine_similarity(toxic_mean.unsqueeze(0), gsm8k_mean.unsqueeze(0)).item():.4f}")

# ── Per-dimension analysis ────────────────────────────────────────────────────
abs_diff = diff.abs()

print(f"\n--- Difference vector stats (across {diff.shape[0]} dims) ---")
print(f"  Mean abs diff : {abs_diff.mean().item():.6f}")
print(f"  Std abs diff  : {abs_diff.std().item():.6f}")
print(f"  Min diff      : {diff.min().item():.6f}  (dim {diff.argmin().item()})")
print(f"  Max diff      : {diff.max().item():.6f}  (dim {diff.argmax().item()})")

TOP_K = 20
top_vals, top_dims = abs_diff.topk(TOP_K)
print(f"\n--- Top {TOP_K} dimensions by |toxic_mean - gsm8k_mean| ---")
print(f"{'Dim':>6}  {'|Diff|':>10}  {'Toxic mean':>12}  {'GSM8K mean':>12}  {'Raw diff':>12}")
for dim, val in zip(top_dims.tolist(), top_vals.tolist()):
    print(f"{dim:>6}  {val:>10.6f}  {toxic_mean[dim].item():>12.6f}  {gsm8k_mean[dim].item():>12.6f}  {diff[dim].item():>12.6f}")

# Percentile breakdown of absolute differences
print("\n--- |Diff| percentiles ---")
for p in [50, 75, 90, 95, 99]:
    v = abs_diff.kthvalue(int(p / 100 * abs_diff.shape[0])).values.item()
    print(f"  p{p:>2}: {v:.6f}")

torch.save({
    "toxic_mean": toxic_mean,
    "gsm8k_mean": gsm8k_mean,
    "diff":       diff,
    "top_dims":   top_dims,
    "top_abs_diffs": top_vals,
}, "/scratch/scholar/ngetach/mean_vectors.pt")
print("\nSaved to /scratch/scholar/ngetach/mean_vectors.pt")
