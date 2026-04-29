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
parser.add_argument("--subspace",               default="/scratch/scholar/ngetach/subspace_toxic_64.pt")
parser.add_argument("--dataset",                default="allenai/real-toxicity-prompts")
parser.add_argument("--split",                  default="train")
parser.add_argument("--prompt_column",          default="prompt")
parser.add_argument("--prompt_subfield",        default="text",
                    help="Subfield to extract if prompt column is a nested dict")
parser.add_argument("--continuation_column",    default="continuation")
parser.add_argument("--continuation_subfield",  default="text",
                    help="Subfield to extract if continuation column is a nested dict")
parser.add_argument("--max_length",             type=int, default=512)
parser.add_argument("--limit",                  type=int, default=None)
parser.add_argument("--output",                 default="/scratch/scholar/ngetach/ppl_eval_results.pt")
args = parser.parse_args()


def get_text(row, column, subfield):
    val = row[column]
    if subfield and isinstance(val, dict):
        return val[subfield]
    return val


def continuation_perplexity(model, tokenizer, prompt_text, continuation_text, device):
    prompt_len = tokenizer(
        prompt_text, return_tensors="pt", truncation=True, max_length=args.max_length
    )["input_ids"].shape[1]

    enc = tokenizer(
        prompt_text + continuation_text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    input_ids      = enc["input_ids"].long().to(device)
    attention_mask = enc["attention_mask"].long().to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    labels = input_ids.clone()
    labels[:, :prompt_len]    = -100  # mask prompt tokens
    labels[attention_mask == 0] = -100  # mask padding

    shift_logits = logits[:, :-1].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()

    n_valid = (shift_labels != -100).sum().item()
    if n_valid == 0:
        return None

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )
    return torch.exp(loss).item()


def evaluate_perplexity(model, tokenizer, dataset, device, label):
    perplexities = []
    for i, row in enumerate(dataset):
        prompt_text = get_text(row, args.prompt_column, args.prompt_subfield)
        cont_text   = get_text(row, args.continuation_column, args.continuation_subfield)
        if not prompt_text or not cont_text:
            continue
        ppl = continuation_perplexity(model, tokenizer, prompt_text, cont_text, device)
        if ppl is not None:
            perplexities.append(ppl)
        if (i + 1) % 50 == 0:
            mean_so_far = sum(perplexities) / len(perplexities) if perplexities else float("nan")
            print(f"  [{label}] {i+1}/{len(dataset)} — running mean PPL: {mean_so_far:.2f}", flush=True)
    return perplexities


# ── load tokenizer & model ───────────────────────────────────────────────────
print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, token=HF_TOKEN, cache_dir=HF_CACHE, trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...", flush=True)
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
dataset = load_dataset(args.dataset, split=args.split, token=HF_TOKEN, streaming=True)
if args.limit:
    dataset = dataset.take(args.limit)
dataset = list(dataset)
print(f"Dataset size: {len(dataset)}", flush=True)

# ── evaluate original model ──────────────────────────────────────────────────
print("\n=== Evaluating ORIGINAL model ===", flush=True)
orig_ppls = evaluate_perplexity(model, tokenizer, dataset, device, "original")
orig_mean = sum(orig_ppls) / len(orig_ppls) if orig_ppls else float("nan")
orig_std  = (sum((x - orig_mean) ** 2 for x in orig_ppls) / len(orig_ppls)) ** 0.5 if orig_ppls else float("nan")
print(f"Original mean PPL: {orig_mean:.2f} ± {orig_std:.2f}", flush=True)

# ── load subspace & patch ff_out ─────────────────────────────────────────────
print(f"\nLoading subspace from {args.subspace}...", flush=True)
subspace = torch.load(args.subspace, weights_only=False)
P_perp   = subspace["P_perp"].to(torch.bfloat16).to(device)
k        = subspace["k"]
ev       = subspace["explained_variance"]
print(f"Subspace: k={k}, explained variance={ev:.4f}", flush=True)

print("Patching ff_out weight: W_new = W @ P_perp ...", flush=True)
ff_out = model.model.transformer.ff_out
with torch.no_grad():
    ff_out.weight.copy_(ff_out.weight @ P_perp)
del P_perp, subspace
torch.cuda.empty_cache()
print("Patch applied.", flush=True)

# ── evaluate patched model ───────────────────────────────────────────────────
print("\n=== Evaluating PATCHED model ===", flush=True)
patched_ppls = evaluate_perplexity(model, tokenizer, dataset, device, "patched")
patched_mean = sum(patched_ppls) / len(patched_ppls) if patched_ppls else float("nan")
patched_std  = (sum((x - patched_mean) ** 2 for x in patched_ppls) / len(patched_ppls)) ** 0.5 if patched_ppls else float("nan")
print(f"Patched mean PPL: {patched_mean:.2f} ± {patched_std:.2f}", flush=True)

# ── summary ──────────────────────────────────────────────────────────────────
delta = patched_mean - orig_mean
print(f"\n=== Summary ===")
print(f"  Original PPL : {orig_mean:.2f} ± {orig_std:.2f}")
print(f"  Patched PPL  : {patched_mean:.2f} ± {patched_std:.2f}")
print(f"  Delta        : {delta:+.2f}  ({'higher = patch suppressed toxic continuations' if delta > 0 else 'lower or no effect'})")

torch.save({
    "orig_perplexities":    orig_ppls,
    "patched_perplexities": patched_ppls,
    "orig_mean_ppl":        orig_mean,
    "orig_std_ppl":         orig_std,
    "patched_mean_ppl":     patched_mean,
    "patched_std_ppl":      patched_std,
    "delta_mean_ppl":       delta,
    "k":                    k,
    "explained_variance":   ev,
}, args.output)
print(f"\nResults saved to {args.output}")
