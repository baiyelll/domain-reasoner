# TRAIN.md – Supervised Fine-Tuning with Unsloth + QLoRA

## Goal
Produce one or more high-quality LoRA adapters for your domain.

---

## 1. Configure Training

Edit `scripts/train_sft.py` and change the following at the top:

```python
MODEL_NAME = "unsloth/Llama-3.1-8B-Instruct"   # or Qwen2.5-7B-Instruct
DATASET_PATH = "data/your_dataset.jsonl"       # or HF dataset name (must be cached)
OUTPUT_DIR = "adapters/domain-v1"
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 32
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
NUM_EPOCHS = 2
LEARNING_RATE = 2e-4
```

---

## 2. Submit Training Job

In the COMPUTE web UI → **SUBMIT JOB**:

- **Name**: `sft-domain-v1`
- **CPU required**: 4–8
- **Memory required**: 32GB–64GB
- **GPU required**: 1
- **GPU Model**: RTX 3090 (or any ≥24GB)
- **Project Directory**: select your project
- **Image**: `yourusername/domain-reasoner:1.0`
- **Command to run**:

```bash
HOME=/tmp; /project/YOUR_PROJECT_NAME/domain-reasoner/scripts/train.sh
```

- Optional: check “Send reminder when job status is changed”

Click **Queue**.

---

## 3. Monitor

- Go to **RECENT JOBS** → click the job ID
- Use **Get Stdout Logs** / **Get Stderr Logs**
- Training progress (loss, steps) will appear there

Typical times on RTX 3090 (24GB):
- 1k–2k examples, 2 epochs → 30–90 minutes
- 5k–10k examples → 2–6 hours

---

## 4. After Training Finishes

Adapters are saved to:

```
adapters/domain-v1/
├── adapter_config.json
├── adapter_model.safetensors   # or .bin
└── ...
```

You can now use this adapter with vLLM (see SERVE.md).

---

## Multiple Domain Adapters

Just change `OUTPUT_DIR` and re-run.  
Example structure:

```
adapters/
├── math-reasoner-v1/
├── medical-guidelines-v1/
└── legal-contract-v1/
```

All of them can later be loaded together on a single vLLM instance.