"""
Computes statistics from already-saved PCA eval .pt files.

Per-example losses were not saved in the original run (only scalar means),
so loss std/stderr are not recoverable. Rank statistics are fully computable
since orig_ranks and patched_ranks tensors were saved.
"""
import torch

FILES = {
    8:   "/scratch/scholar/ngetach/eval_results_pca_toxic_final_k8.pt",
    32:  "/scratch/scholar/ngetach/eval_results_pca_toxic_final_k32.pt",
    64:  "/scratch/scholar/ngetach/eval_results_pca_toxic_final_k64.pt",
    128: "/scratch/scholar/ngetach/eval_results_pca_toxic_final_k128.pt",
}

print(f"{'k':>4}  {'Orig Loss':>10}  {'Patch Loss':>10}  {'Δ Loss':>8}  "
      f"{'Orig Rank Mean':>15}  {'Orig Rank Std':>14}  {'Orig Rank SE':>13}  "
      f"{'Patch Rank Mean':>16}  {'Patch Rank Std':>15}  {'Patch Rank SE':>14}")
print("-" * 145)

for k, path in sorted(FILES.items()):
    r = torch.load(path, weights_only=False)

    orig_loss    = r["original_loss"]
    patched_loss = r["patched_loss"]
    delta        = r["delta"]

    orig_ranks    = r["orig_ranks"].float()
    patched_ranks = r["patched_ranks"].float()

    orig_rank_mean = orig_ranks.mean().item()
    orig_rank_std  = orig_ranks.std().item()
    orig_rank_se   = orig_rank_std / (orig_ranks.numel() ** 0.5)

    pat_rank_mean  = patched_ranks.mean().item()
    pat_rank_std   = patched_ranks.std().item()
    pat_rank_se    = pat_rank_std / (patched_ranks.numel() ** 0.5)

    print(f"{k:>4}  {orig_loss:>10.4f}  {patched_loss:>10.4f}  {delta:>+8.4f}  "
          f"{orig_rank_mean:>15.1f}  {orig_rank_std:>14.1f}  {orig_rank_se:>13.4f}  "
          f"{pat_rank_mean:>16.1f}  {pat_rank_std:>15.1f}  {pat_rank_se:>14.4f}")

print()
print("Note: loss std/stderr are not available from these files (only scalar means were saved).")
print("      Rerun with the updated patch_and_eval.py to get per-example loss distributions.")
