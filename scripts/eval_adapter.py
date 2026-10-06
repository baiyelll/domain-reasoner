#!/usr/bin/env python3
"""
Simple offline evaluation of a trained adapter.
Useful for quick sanity checks inside a COMPUTE job.
"""

from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch

MODEL_NAME = "unsloth/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "adapters/domain-v1"
MAX_SEQ_LENGTH = 2048

print("Loading base + adapter...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, adapter_name=ADAPTER_PATH)  # or load properly
FastLanguageModel.for_inference(model)

# Example prompts – replace with your domain questions
prompts = [
    "Explain the key steps in solving a complex domain problem.",
    "What are the main considerations when analyzing this type of data?",
]

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

for prompt in prompts:
    print("\n" + "="*60)
    print("PROMPT:", prompt)
    print("-"*60)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    _ = model.generate(**inputs, max_new_tokens=256, streamer=streamer, temperature=0.7)