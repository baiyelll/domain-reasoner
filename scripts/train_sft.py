#!/usr/bin/env python3
"""
Domain-Specific Reasoner – Unsloth QLoRA SFT script
Optimized for HKUST COMPUTE (offline, single GPU).
"""

import os
import torch
from pathlib import Path
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# ============================================================
# CONFIG – edit these
# ============================================================
MODEL_NAME = "unsloth/Llama-3.1-8B-Instruct"          # or local path under models/
DATASET_PATH = "data/your_dataset.jsonl"              # jsonl or HF dataset name
OUTPUT_DIR = "adapters/domain-v1"

MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 2
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_STEPS = 100
SEED = 42

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

# Prefer local copy if available
local_model = PROJECT_ROOT / "models" / MODEL_NAME.replace("/", "--")
model_path = str(local_model) if local_model.exists() else MODEL_NAME

print(f"Loading model from: {model_path}")

# ============================================================
# Load model with Unsloth (4-bit)
# ============================================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,          # auto
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)

# ============================================================
# Dataset
# ============================================================
if DATASET_PATH.endswith((".json", ".jsonl")):
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
else:
    dataset = load_dataset(DATASET_PATH, split="train")

# Simple formatting function (ChatML style)
def formatting_prompts_func(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}

# If your data uses "messages" key
if "messages" in dataset.column_names:
    dataset = dataset.map(formatting_prompts_func, batched=True)
else:
    # Fallback for Alpaca-style
    def alpaca_format(examples):
        texts = []
        for inst, inp, out in zip(examples["instruction"], examples.get("input", [""]*len(examples["instruction"])), examples["output"]):
            if inp:
                prompt = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
            else:
                prompt = f"### Instruction:\n{inst}\n\n### Response:\n{out}"
            texts.append(prompt)
        return {"text": texts}
    dataset = dataset.map(alpaca_format, batched=True)

# ============================================================
# Trainer
# ============================================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=True,          # efficient packing
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=SEED,
        report_to="none",          # change to "wandb" if you want
    ),
)

print("Starting training...")
trainer.train()

# Save final adapter
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR}")