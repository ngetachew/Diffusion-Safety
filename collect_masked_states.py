import os
import argparse
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_CACHE = os.getenv("HF_CACHE")
MODEL_ID = "GSAI-ML/LLaDA-8B-Base"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, help="HuggingFace dataset name")
parser.add_argument("--split", default="train", help="Dataset split")
parser.add_argument("--text_column", default="prompt", help="Column containing prompt text")
parser.add_argument("--text_subfield", default=None, help="If text column is a nested dict, extract this key")
parser.add_argument("--concat_column", default=None, help="Optional second column to concatenate (e.g. continuation)")
parser.add_argument("--concat_subfield", default=None, help="Subfield to extract from the concat column if it is a nested dict")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--output", default="hidden_states_masked.pt", help="Output file path")
parser.add_argument("--limit", type=int, default=None, help="Cap number of examples")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset_config", default=None, help="HuggingFace dataset config name (e.g. 'main' for openai/gsm8k)")
args = parser.parse_args()

torch.manual_seed(args.seed)

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, token=HF_TOKEN, cache_dir=HF_CACHE, trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    cache_dir=HF_CACHE,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

print(f"Loading dataset '{args.dataset}' split='{args.split}'...", flush=True)
dataset = load_dataset(args.dataset, args.dataset_config, split=args.split, token=HF_TOKEN, cache_dir=HF_CACHE, streaming=True)
if args.limit is not None:
    dataset = dataset.take(args.limit)
dataset = list(dataset)
print(f"Dataset size: {len(dataset)}", flush=True)

first_param = next(model.parameters())
input_device = first_param.device

# LLaDA's tokenizer doesn't expose mask_token_id directly; fall back to model config.
mask_token_id = tokenizer.mask_token_id
if mask_token_id is None:
    mask_token_id = model.config.mask_token_id
if mask_token_id is None:
    raise ValueError("Could not determine mask_token_id from tokenizer or model config.")

all_hidden_states = []
all_texts = []
all_mask_positions = []
all_mask_ratios = []

def sample_mask_positions(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each example, sample a mask ratio uniformly in [0, 1],
    then mask each valid token independently with that probability.

    Returns:
        masked: [B, T] boolean mask
        ratios: [B] sampled mask ratios
    """
    device = attention_mask.device
    valid = attention_mask.bool()  # [B, T]
    B, T = valid.shape

    ratios = torch.rand(B, device=device)  # Uniform(0,1) per example
    rand = torch.rand(B, T, device=device)

    masked = rand < ratios.unsqueeze(1)
    masked = masked & valid

    valid_counts = valid.sum(dim=1)
    masked_counts = masked.sum(dim=1)

    # Ensure at least one masked token for every non-empty example
    for b in range(B):
        if valid_counts[b] > 0 and masked_counts[b] == 0:
            valid_indices = torch.nonzero(valid[b], as_tuple=False).squeeze(-1)
            chosen = valid_indices[torch.randint(len(valid_indices), (1,), device=device)]
            masked[b, chosen] = True

    return masked, ratios

for start in range(0, len(dataset), args.batch_size):
    batch = dataset[start : start + args.batch_size]
    texts = [row[args.text_column] for row in batch]
    if args.text_subfield:
        texts = [t[args.text_subfield] if isinstance(t, dict) else t for t in texts]

    if args.concat_column:
        continuations = [row[args.concat_column] for row in batch]
        if args.concat_subfield:
            continuations = [c[args.concat_subfield] if isinstance(c, dict) else c for c in continuations]
        texts = [p + (c or "") for p, c in zip(texts, continuations)]

    texts = [t for t in texts if t and t.strip()]
    if not texts:
        continue

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )
    input_ids = inputs["input_ids"].long().to(input_device)
    attention_mask = inputs["attention_mask"].long().to(input_device)

    if input_ids.shape[1] == 0:
        continue

    masked_pos, mask_ratios = sample_mask_positions(attention_mask)
    corrupted_input_ids = input_ids.clone()
    corrupted_input_ids[masked_pos] = mask_token_id

    with torch.no_grad():
        outputs = model(
            input_ids=corrupted_input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden = outputs.hidden_states[-1].detach().cpu().to(torch.float32)  # [B, T, 4096]
    masked_pos_cpu = masked_pos.detach().cpu()
    mask_ratios_cpu = mask_ratios.detach().cpu()

    masked_hidden = hidden[masked_pos_cpu]  # [N_masked, 4096]
    if masked_hidden.numel() == 0:
        continue

    all_hidden_states.append(masked_hidden)

    batch_indices, token_indices = torch.nonzero(masked_pos_cpu, as_tuple=True)
    for b_idx, t_idx in zip(batch_indices.tolist(), token_indices.tolist()):
        all_texts.append(texts[b_idx])
        all_mask_positions.append(t_idx)
        all_mask_ratios.append(mask_ratios_cpu[b_idx].item())

    if (start // args.batch_size) % 10 == 0:
        print(
            f"  Processed {min(start + args.batch_size, len(dataset))}/{len(dataset)} "
            f"| collected {sum(x.size(0) for x in all_hidden_states)} masked states"
        )

if not all_hidden_states:
    raise RuntimeError("No masked hidden states were collected.")

all_hidden_states = torch.cat(all_hidden_states, dim=0)  # [N, 4096]
print(f"Masked hidden states shape: {all_hidden_states.shape}")

torch.save(
    {
        "hidden_states": all_hidden_states,
        "texts": all_texts,
        "mask_positions": all_mask_positions,
        "mask_ratios": all_mask_ratios,
        "model_id": MODEL_ID,
    },
    args.output,
)
print(f"Saved to {args.output}")