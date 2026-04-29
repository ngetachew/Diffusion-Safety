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
parser.add_argument("--dataset", required=True, help="HuggingFace dataset name, e.g. 'PKU-Alignment/PKU-SafeRLHF'")
parser.add_argument("--split", default="train", help="Dataset split")
parser.add_argument("--text_column", default="prompt", help="Column name containing the prompt text")
parser.add_argument("--text_subfield", default=None, help="If the text column is a nested dict, extract this key (e.g. 'text')")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--output", default="hidden_states.pt", help="Output file path")
parser.add_argument("--limit", type=int, default=None, help="Cap number of examples (useful for testing)")
args = parser.parse_args()

print(f"Loading tokenizer and model...")
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

# Hook into ln_f to capture pre-projection hidden states (shape: [batch, seq_len, 4096])
_hook_buffer = []
def _hook_fn(module, input, output):
    _hook_buffer.append(output.detach().cpu().to(torch.float32))

hook = model.model.transformer.ln_f.register_forward_hook(_hook_fn)

print(f"Loading dataset '{args.dataset}' split='{args.split}'...", flush=True)
dataset = load_dataset(args.dataset, split=args.split, token=HF_TOKEN, streaming=True)
if args.limit is not None:
    dataset = dataset.take(args.limit)
dataset = list(dataset)
print(f"Dataset size: {len(dataset)}", flush=True)

# Determine the input device (first device used by the model)
first_param = next(model.parameters())
input_device = first_param.device

all_hidden_states = []
all_texts = []

for start in range(0, len(dataset), args.batch_size):
    batch = dataset[start : start + args.batch_size]
    texts = [row[args.text_column] for row in batch]
    if args.text_subfield:
        texts = [t[args.text_subfield] if isinstance(t, dict) else t for t in texts]

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

    _hook_buffer.clear()
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)

    # _hook_buffer has exactly one entry: [batch, seq_len, 4096]
    # (clear before forward to discard any stale entries from prior calls)
    hidden = _hook_buffer[-1]  # CPU float32
    _hook_buffer.clear()

    # Index of the last real (non-padding) token for each sequence
    last_indices = attention_mask.sum(dim=1) - 1  # [batch], on GPU
    last_indices = last_indices.cpu()

    last_hidden = hidden[torch.arange(hidden.size(0)), last_indices]  # [batch, 4096]
    all_hidden_states.append(last_hidden)
    all_texts.extend(texts)

    if (start // args.batch_size) % 10 == 0:
        print(f"  Processed {min(start + args.batch_size, len(dataset))}/{len(dataset)}")

hook.remove()

all_hidden_states = torch.cat(all_hidden_states, dim=0)  # [N, 4096]
print(f"Hidden states shape: {all_hidden_states.shape}")

torch.save({"hidden_states": all_hidden_states, "texts": all_texts}, args.output)
print(f"Saved to {args.output}")
