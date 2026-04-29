"""
Patches LLaDA's ff_out weight using the difference-of-means direction as U,
then evaluates both the original and patched model on a HuggingFace dataset.

Instead of PCA, the rank-1 subspace basis is:
    u = (mean_toxic - mean_benign) / ||mean_toxic - mean_benign||

The projection matrix is:
    P_perp = I - u @ u.T

Applied as:
    ff_out_new.weight = ff_out.weight @ P_perp

so that the component of any hidden state along the toxic-vs-benign
discrimination direction is zeroed out before projection to vocab.
"""
import gc
import os
import argparse
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_CACHE  = os.getenv("HF_CACHE")
MODEL_ID  = "GSAI-ML/LLaDA-8B-Base"

parser = argparse.ArgumentParser()
parser.add_argument("--toxic_states",  required=True,
                    help="Path to toxic hidden states .pt file")
parser.add_argument("--benign_states", required=True,
                    help="Path to benign hidden states .pt file")
parser.add_argument("--dataset",       default="P1ayer-1/books-3-textbooks")
parser.add_argument("--dataset_config", default=None,
                    help="HuggingFace dataset config name (e.g. 'main' for openai/gsm8k)")
parser.add_argument("--split",         default="train")
parser.add_argument("--text_column",   default="text")
parser.add_argument("--text_subfield", default=None)
parser.add_argument("--continuation_column",   default=None)
parser.add_argument("--continuation_subfield", default=None)
parser.add_argument("--batch_size",    type=int, default=4)
parser.add_argument("--max_length",    type=int, default=512)
parser.add_argument("--limit",         type=int, default=None)
parser.add_argument("--renormalize",   action="store_true",
                    help="Scale patched weight back to original Frobenius norm after projection")
parser.add_argument("--output",        default="/scratch/scholar/ngetach/eval_results_meandiff.pt")
args = parser.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def token_ranks(shift_logits_2d, shift_labels_1d):
    """Return ranks (1-based) of correct tokens at valid positions.

    shift_logits_2d : [L, V]  float32
    shift_labels_1d : [L]     int64, -100 at ignored positions
    Returns a 1-D CPU int64 tensor of ranks for valid positions.
    """
    valid = shift_labels_1d != -100                          # [L] bool
    if not valid.any():
        return torch.empty(0, dtype=torch.long)
    logits_v = shift_logits_2d[valid]                        # [N, V]
    labels_v = shift_labels_1d[valid]                        # [N]
    correct_logit = logits_v[torch.arange(len(labels_v)), labels_v]  # [N]
    ranks = (logits_v > correct_logit.unsqueeze(1)).sum(dim=1) + 1   # [N]
    return ranks.cpu()


def compute_example(model, tokenizer, prompt_text, continuation_text, device):
    """Return (loss, ranks_tensor) for a single prompt+continuation example."""
    prompt_len = tokenizer(
        prompt_text, return_tensors="pt", truncation=True, max_length=args.max_length
    )["input_ids"].shape[1]
    enc = tokenizer(
        prompt_text + continuation_text,
        return_tensors="pt", truncation=True, max_length=args.max_length,
    )
    input_ids      = enc["input_ids"].long().to(device)
    attention_mask = enc["attention_mask"].long().to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    labels = input_ids.clone()
    labels[:, :prompt_len]      = -100
    labels[attention_mask == 0] = -100
    shift_logits = logits[0, :-1].float()    # [L, V]
    shift_labels = labels[0, 1:]             # [L]
    n_valid = (shift_labels != -100).sum().item()
    if n_valid == 0:
        return None, torch.empty(0, dtype=torch.long)
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    ).item()
    ranks = token_ranks(shift_logits, shift_labels)
    return loss, ranks


def compute_fulltext(model, input_ids, attention_mask):
    """Return (loss, ranks_tensor) for a full-text example."""
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[0, :-1].float()
    shift_labels = input_ids[0, 1:].clone()
    shift_mask   = attention_mask[0, 1:]
    shift_labels[shift_mask == 0] = -100
    n_valid = (shift_labels != -100).sum().item()
    if n_valid == 0:
        return 0.0, torch.empty(0, dtype=torch.long)
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    ).item()
    ranks = token_ranks(shift_logits, shift_labels)
    return loss, ranks


def evaluate(model, tokenizer, dataset, device):
    """Returns (losses, all_ranks) where:
      losses    : float32 tensor of per-example losses  [N]
      all_ranks : int64 tensor of per-token ranks       [M]
    """
    losses    = []
    all_ranks = []
    n_valid   = 0

    for row in dataset:
        prompt = row[args.text_column]
        if args.text_subfield:
            prompt = prompt[args.text_subfield] if isinstance(prompt, dict) else prompt
        if not prompt or not prompt.strip():
            continue

        if args.continuation_column:
            cont = row[args.continuation_column]
            if args.continuation_subfield:
                cont = cont[args.continuation_subfield] if isinstance(cont, dict) else cont
            if not cont or not cont.strip():
                continue
            loss_val, ranks = compute_example(model, tokenizer, prompt, cont, device)
        else:
            inputs = tokenizer(
                [prompt], return_tensors="pt", padding=True,
                truncation=True, max_length=args.max_length,
            )
            input_ids      = inputs["input_ids"].long().to(device)
            attention_mask = inputs["attention_mask"].long().to(device)
            loss_val, ranks = compute_fulltext(model, input_ids, attention_mask)

        if loss_val is None:
            continue
        losses.append(loss_val)
        n_valid += 1
        if ranks.numel():
            all_ranks.append(ranks)
        if n_valid % 10 == 0:
            running_mean = sum(losses) / n_valid
            print(f"  {n_valid}/{len(dataset)}, running loss {running_mean:.4f}", flush=True)

    losses    = torch.tensor(losses) if losses else torch.empty(0)
    all_ranks = torch.cat(all_ranks) if all_ranks else torch.empty(0, dtype=torch.long)
    return losses, all_ranks


def loss_stats(losses):
    """Return (mean, std, stderr) for a 1-D loss tensor."""
    n    = losses.numel()
    mean = losses.mean().item()
    std  = losses.std().item()           # Bessel-corrected (ddof=1)
    se   = std / (n ** 0.5)
    return mean, std, se


# ── build rank-1 projection from mean difference ─────────────────────────────
print(f"Loading toxic hidden states from {args.toxic_states}...")
toxic_data  = torch.load(args.toxic_states,  weights_only=False)
toxic_mean  = toxic_data["hidden_states"].float().mean(dim=0)   # [4096]
print(f"  Toxic  : {toxic_data['hidden_states'].shape}, mean norm={toxic_mean.norm():.4f}")
del toxic_data

print(f"Loading benign hidden states from {args.benign_states}...")
benign_data = torch.load(args.benign_states, weights_only=False)
benign_mean = benign_data["hidden_states"].float().mean(dim=0)  # [4096]
print(f"  Benign : {benign_data['hidden_states'].shape}, mean norm={benign_mean.norm():.4f}")
del benign_data

diff = toxic_mean - benign_mean                          # [4096]
u    = (diff / diff.norm()).unsqueeze(1)                 # [4096, 1]  unit column vector
print(f"Diff norm: {diff.norm().item():.4f}")
print(f"Cosine similarity of means: {F.cosine_similarity(toxic_mean.unsqueeze(0), benign_mean.unsqueeze(0)).item():.4f}")

# ── load tokenizer & model ───────────────────────────────────────────────────
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, token=HF_TOKEN, cache_dir=HF_CACHE, trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    cache_dir=HF_CACHE,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
device = next(model.parameters()).device
n_gpus = torch.cuda.device_count()
print(f"GPUs visible: {n_gpus}")
for i in range(n_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GiB", flush=True)

# ── load dataset ─────────────────────────────────────────────────────────────
print(f"\nLoading dataset '{args.dataset}'...", flush=True)
dataset = load_dataset(
    args.dataset, args.dataset_config,
    split=args.split, token=HF_TOKEN, cache_dir=HF_CACHE, streaming=True,
)
if args.limit:
    dataset = dataset.take(args.limit)
dataset = list(dataset)
print(f"Dataset size: {len(dataset)}", flush=True)

# ── evaluate original model ──────────────────────────────────────────────────
print("\n=== Evaluating ORIGINAL model ===")
orig_losses, orig_ranks = evaluate(model, tokenizer, dataset, device)
orig_mean, orig_std, orig_se = loss_stats(orig_losses)
orig_rank1 = (orig_ranks == 1).sum().item()
print(f"Original mean loss : {orig_mean:.4f} ± {orig_std:.4f}  (SE={orig_se:.4f})")
print(f"Original rank-1    : {orig_rank1} / {orig_ranks.numel()} tokens "
      f"({100*orig_rank1/max(orig_ranks.numel(),1):.2f}%)", flush=True)

# Free activations cached during eval before patching
gc.collect()
torch.cuda.empty_cache()

# ── patch ff_out ─────────────────────────────────────────────────────────────
# Instead of modifying the weight matrix (which fails when accelerate offloads
# the layer to CPU and makes ff_out.weight a read-only meta placeholder), we
# register a forward_pre_hook that projects the hidden state before it reaches
# ff_out. This is mathematically identical to the weight patch:
#
#   h @ W_new.T  =  (h - (h·u)u.T) @ W.T
#
# The hook is device-agnostic: it moves u to whatever device the input arrives
# on, so it works regardless of how device_map="auto" places the layer.
print("\nRegistering mean-diff projection hook on ff_out...", flush=True)

u_flat = u.squeeze(1).float()   # [4096] unit vector, on CPU

def _mean_diff_hook(module, inputs):
    h = inputs[0]                                        # [B, T, 4096] or [B, 4096]
    u_loc = u_flat.to(dtype=h.dtype, device=h.device)   # match input device/dtype
    proj = (h @ u_loc).unsqueeze(-1) * u_loc            # [B, T, 4096]
    return (h - proj,) + inputs[1:]

ff_out = model.model.transformer.ff_out
ff_out.register_forward_pre_hook(_mean_diff_hook)
print(f"  Projection direction norm: {u_flat.norm().item():.6f}  (should be 1.0)", flush=True)
print(f"  Diff vector norm: {diff.norm().item():.4f}", flush=True)
print("Patch applied.")

# ── evaluate patched model ───────────────────────────────────────────────────
print("\n=== Evaluating PATCHED model ===")
patched_losses, patched_ranks = evaluate(model, tokenizer, dataset, device)
patched_mean, patched_std, patched_se = loss_stats(patched_losses)
patched_rank1 = (patched_ranks == 1).sum().item()
print(f"Patched mean loss  : {patched_mean:.4f} ± {patched_std:.4f}  (SE={patched_se:.4f})")
print(f"Patched rank-1     : {patched_rank1} / {patched_ranks.numel()} tokens "
      f"({100*patched_rank1/max(patched_ranks.numel(),1):.2f}%)", flush=True)

# ── summary ──────────────────────────────────────────────────────────────────
delta       = patched_mean  - orig_mean
delta_rank1 = patched_rank1 - orig_rank1
n_tokens    = orig_ranks.numel()

print(f"\n=== Summary ===")
print(f"  Original loss  : {orig_mean:.4f} ± {orig_std:.4f}  (SE={orig_se:.4f})")
print(f"  Patched  loss  : {patched_mean:.4f} ± {patched_std:.4f}  (SE={patched_se:.4f})")
print(f"  Δ loss         : {delta:+.4f}")
print(f"  Original rank-1: {orig_rank1} / {n_tokens}  ({100*orig_rank1/max(n_tokens,1):.2f}%)")
print(f"  Patched  rank-1: {patched_rank1} / {n_tokens}  ({100*patched_rank1/max(n_tokens,1):.2f}%)")
print(f"  Δ rank-1       : {delta_rank1:+d} tokens")
print(f"  Mean rank (orig)   : {orig_ranks.float().mean().item():.1f}")
print(f"  Mean rank (patched): {patched_ranks.float().mean().item():.1f}")

torch.save({
    "orig_losses":      orig_losses,
    "patched_losses":   patched_losses,
    "orig_mean":        orig_mean,
    "patched_mean":     patched_mean,
    "orig_std":         orig_std,
    "patched_std":      patched_std,
    "orig_se":          orig_se,
    "patched_se":       patched_se,
    "delta":            delta,
    "orig_ranks":       orig_ranks,
    "patched_ranks":    patched_ranks,
    "orig_rank1":       orig_rank1,
    "patched_rank1":    patched_rank1,
    "diff_norm":        diff.norm().item(),
    "toxic_states":     args.toxic_states,
    "benign_states":    args.benign_states,
}, args.output)
print(f"\nResults saved to {args.output}")
