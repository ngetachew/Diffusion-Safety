import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_CACHE = os.getenv("HF_CACHE")

MODEL_ID = "GSAI-ML/LLaDA-8B-Base"

print(f"Cache dir: {HF_CACHE}")
print(f"Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    cache_dir=HF_CACHE,
    trust_remote_code=True,
)

print(f"Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    cache_dir=HF_CACHE,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print(f"Model loaded successfully.")
print(f"Model type: {type(model)}")
print(f"Model config: {model.config}")
print(f"\nModel architecture:")
print(model)
