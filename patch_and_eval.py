"""
Patches LLaDA's ff_out weight by right-multiplying with P_perp = I - UU^T,
then evaluates both the original and patched model on a HuggingFace dataset.

The modification bakes the projection directly into the weight matrix:
    ff_out_new.weight = ff_out.weight @ P_perp   ([126464, 4096] @ [4096, 4096])

so that for any hidden state h:
    ff_out_new(h) = ff_out(P_perp @ h)

Evaluation metric: mean cross-entropy loss over all token positions
(treats the model as a standard LM — a reasonable proxy for reconstruction quality).
"""
import os
import copy
import argparse
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch.nn.functional as F

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_CACHE  = os.getenv("HF_CACHE")
MODEL_ID  = "GSAI-ML/LLaDA-8B-Base"

parser = argparse.ArgumentParser()
parser.add_argument("--subspace",      default="/scratch/scholar/ngetach/subspace.pt")
parser.add_argument("--dataset",       default="P1ayer-1/books-3-textbooks")
parser.add_argument("--split",         default="train")
parser.add_argument("--dataset_config", default=None,
                    help="HuggingFace dataset config name (e.g. 'main' for openai/gsm8k)")
parser.add_argument("--text_column",   default="text")
parser.add_argument("--text_subfield",          default=None,
                    help="If the text column is a nested dict, extract this key (e.g. 'text')")
parser.add_argument("--continuation_column",    default=None,
                    help="If set, compute loss only on continuation tokens (masks prompt in labels)")
parser.add_argument("--continuation_subfield",  default=None,
                    help="Subfield to extract from the continuation column if it is a nested dict")
parser.add_argument("--batch_size",    type=int, default=4)
parser.add_argument("--max_length",    type=int, default=512)
parser.add_argument("--limit",         type=int, default=None)
parser.add_argument("--renormalize",   action="store_true",
                    help="Scale patched weight back to original Frobenius norm after projection")
parser.add_argument("--output",        default="/scratch/scholar/ngetach/eval_results.pt")
args = parser.parse_args()


# ── helpers ──────────────────────────────────────────────────────────────────

def token_ranks(shift_logits_2d, shift_labels_1d):
    """Return 1-based ranks of correct tokens at valid positions.

    shift_logits_2d : [L, V]  float32
    shift_labels_1d : [L]     int64, -100 at ignored positions
    Returns a 1-D CPU int64 tensor of ranks for valid positions.
    """
    valid = shift_labels_1d != -100
    if not valid.any():
        return torch.empty(0, dtype=torch.long)
    logits_v = shift_logits_2d[valid]                                         # [N, V]
    labels_v = shift_labels_1d[valid]                                         # [N]
    correct_logit = logits_v[torch.arange(len(labels_v)), labels_v]           # [N]
    ranks = (logits_v > correct_logit.unsqueeze(1)).sum(dim=1) + 1            # [N]
    return ranks.cpu()


def compute_loss(model, input_ids, attention_mask, debug=False):
    """Mean cross-entropy and per-token ranks over non-padding positions."""
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    if debug:
        print(f"  [debug] logits shape={logits.shape} dtype={logits.dtype} "
              f"nan%={logits.isnan().float().mean().item():.4f} "
              f"inf%={logits.isinf().float().mean().item():.4f}")

    shift_logits = logits[0, :-1].float()
    shift_labels = input_ids[0, 1:].clone()
    shift_mask   = attention_mask[0, 1:]
    shift_labels[shift_mask == 0] = -100

    n_valid = (shift_labels != -100).sum().item()
    if debug:
        print(f"  [debug] n_valid_labels={n_valid} / {shift_labels.numel()}")

    if n_valid == 0:
        return torch.tensor(0.0), torch.empty(0, dtype=torch.long)

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )
    ranks = token_ranks(shift_logits, shift_labels)
    return loss, ranks


def compute_continuation_loss(model, tokenizer, prompt_text, continuation_text, device):
    """Loss and per-token ranks computed only on continuation tokens."""
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
    shift_logits = logits[0, :-1].float()
    shift_labels = labels[0, 1:]
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


def evaluate(model, tokenizer, dataset, device):
    """Returns (mean_loss, all_ranks) where all_ranks is a 1-D tensor of per-token ranks."""
    total_loss = 0.0
    n_valid    = 0
    all_ranks  = []

    for i, row in enumerate(dataset):
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
            loss_val, ranks = compute_continuation_loss(model, tokenizer, prompt, cont, device)
        else:
            inputs = tokenizer(
                [prompt], return_tensors="pt", padding=True,
                truncation=True, max_length=args.max_length,
            )
            input_ids      = inputs["input_ids"].long().to(device)
            attention_mask = inputs["attention_mask"].long().to(device)
            loss_val, ranks = compute_loss(model, input_ids, attention_mask)
            loss_val = loss_val.item()

        if loss_val is None:
            continue
        total_loss += loss_val
        n_valid    += 1
        if ranks.numel():
            all_ranks.append(ranks)

        if n_valid % 10 == 0:
            print(f"  {n_valid}/{len(dataset)}, running loss {total_loss/n_valid:.4f}", flush=True)

    all_ranks = torch.cat(all_ranks) if all_ranks else torch.empty(0, dtype=torch.long)
    return (total_loss / n_valid if n_valid else float("nan")), all_ranks


# ── load tokenizer & model ───────────────────────────────────────────────────
print("Loading tokenizer...")
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

# ── load dataset ─────────────────────────────────────────────────────────────
print(f"Loading dataset '{args.dataset}'...", flush=True)
dataset = load_dataset(args.dataset, args.dataset_config, split=args.split, token=HF_TOKEN, cache_dir=HF_CACHE, streaming=True)
if args.limit:
    dataset = dataset.take(args.limit)
dataset = list(dataset)
print(f"Dataset size: {len(dataset)}", flush=True)

# ── evaluate original model ──────────────────────────────────────────────────
print("\n=== Evaluating ORIGINAL model ===")
orig_loss, orig_ranks = evaluate(model, tokenizer, dataset, device)
orig_rank1 = (orig_ranks == 1).sum().item()
print(f"Original mean loss : {orig_loss:.4f}")
print(f"Original rank-1    : {orig_rank1} / {orig_ranks.numel()} tokens "
      f"({100*orig_rank1/max(orig_ranks.numel(),1):.2f}%)", flush=True)

# ── patch ff_out ─────────────────────────────────────────────────────────────
print(f"\nLoading subspace from {args.subspace}...")
subspace = torch.load(args.subspace, weights_only=False)
P_perp   = subspace["P_perp"].to(torch.bfloat16).to(device)  # [4096, 4096]
k        = subspace["k"]
ev       = subspace["explained_variance"]
print(f"Subspace: k={k}, explained variance={ev:.4f}")

print("Patching ff_out weight: W_new = W @ P_perp ...", flush=True)
ff_out = model.model.transformer.ff_out
# If device_map="auto" offloaded this layer to CPU/meta, move it to GPU first.
if ff_out.weight.is_meta or ff_out.weight.device.type == "cpu":
    ff_out = ff_out.to(device)
with torch.no_grad():
    norm_before = ff_out.weight.float().norm().item()
    patched = ff_out.weight @ P_perp
    norm_after = patched.float().norm().item()
    print(f"  ff_out weight norm before: {norm_before:.4f}", flush=True)
    print(f"  ff_out weight norm after:  {norm_after:.4f}  (ratio={norm_after/norm_before:.4f})", flush=True)
    if args.renormalize:
        patched = patched * (norm_before / norm_after)
        print(f"  Renormalized to original norm ({norm_before:.4f})", flush=True)
    ff_out.weight.copy_(patched)

del P_perp, subspace, patched
torch.cuda.empty_cache()
print("Patch applied.")

# ── evaluate patched model ───────────────────────────────────────────────────
print("\n=== Evaluating PATCHED model ===")
patched_loss, patched_ranks = evaluate(model, tokenizer, dataset, device)
patched_rank1 = (patched_ranks == 1).sum().item()
print(f"Patched mean loss  : {patched_loss:.4f}")
print(f"Patched rank-1     : {patched_rank1} / {patched_ranks.numel()} tokens "
      f"({100*patched_rank1/max(patched_ranks.numel(),1):.2f}%)", flush=True)

# ── summary ──────────────────────────────────────────────────────────────────
delta       = patched_loss  - orig_loss
delta_rank1 = patched_rank1 - orig_rank1
n_tokens    = orig_ranks.numel()

print(f"\n=== Summary ===")
print(f"  Original loss  : {orig_loss:.4f}")
print(f"  Patched  loss  : {patched_loss:.4f}")
print(f"  Δ loss         : {delta:+.4f}  ({'higher = model knows less about forbidden content' if delta > 0 else 'lower'})")
print(f"  Original rank-1: {orig_rank1} / {n_tokens}  ({100*orig_rank1/max(n_tokens,1):.2f}%)")
print(f"  Patched  rank-1: {patched_rank1} / {n_tokens}  ({100*patched_rank1/max(n_tokens,1):.2f}%)")
print(f"  Δ rank-1       : {delta_rank1:+d} tokens")
print(f"  Mean rank (orig)   : {orig_ranks.float().mean().item():.1f}")
print(f"  Mean rank (patched): {patched_ranks.float().mean().item():.1f}")

torch.save({
    "original_loss":    orig_loss,
    "patched_loss":     patched_loss,
    "delta":            delta,
    "orig_ranks":       orig_ranks,
    "patched_ranks":    patched_ranks,
    "orig_rank1":       orig_rank1,
    "patched_rank1":    patched_rank1,
    "k":                k,
    "explained_variance": ev,
}, args.output)
print(f"\nResults saved to {args.output}")
