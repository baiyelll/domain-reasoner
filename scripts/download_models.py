#!/usr/bin/env python3
"""
Pre-download base models and tokenizers into the project directory.
Run this once while you still have internet (or inside a setup job).
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Models you plan to use
MODELS = [
    "unsloth/Llama-3.1-8B-Instruct",
    "unsloth/Qwen2.5-7B-Instruct",
    # Add more if needed
]

def download(repo_id: str):
    local_dir = MODELS_DIR / repo_id.replace("/", "--")
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"[skip] {repo_id} already exists at {local_dir}")
        return
    print(f"[download] {repo_id} → {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"[done] {repo_id}")

if __name__ == "__main__":
    for m in MODELS:
        download(m)
    print("\nAll models downloaded. You can now train offline.")