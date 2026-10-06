# SERVE.md – vLLM Multi-LoRA Serving

## Goal

Serve the base model + one or more domain LoRA adapters from a **single GPU instance** using vLLM’s continuous batching and dynamic adapter loading (`--enable-lora`).

This is the inference-engineering half of the project.

> **Note:** vLLM requires a GPU for 8B models. The commands below are ready to run as soon as you have access to a GPU (Colab Pro, Kaggle, local NVIDIA, cloud instance, etc.).

---

## 1. Install vLLM

```bash
pip install vllm
```

(Use a recent version that supports LoRA. On Colab you may need to match CUDA version.)

---

## 2. Basic Serve Command (single adapter)

Assuming your trained adapter is at `./adapters/swe-coding-agent-v1`:

```bash
vllm serve unsloth/Llama-3.1-8B-Instruct \
  --enable-lora \
  --lora-modules coding=./adapters/swe-coding-agent-v1 \
  --max-lora-rank 16 \
  --dtype float16 \
  --max-model-len 2048 \
  --port 8000
```

Key flags:
- `--enable-lora` → turns on dynamic LoRA support
- `--lora-modules coding=./path` → registers an adapter under the name `coding`
- `--max-lora-rank 16` → must be ≥ the rank you trained with (we used 16)

---

## 3. Multi-Adapter (multi-tenant) Example

You can load several domain adapters at once:

```bash
vllm serve unsloth/Llama-3.1-8B-Instruct \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 16 \
  --lora-modules \
      coding=./adapters/swe-coding-agent-v1 \
      general=./adapters/domain-v1 \
  --dtype float16 \
  --max-model-len 2048
```

Clients choose which adapter to use by passing the adapter name as the `model` field.

---

## 4. Calling the Server

### cURL

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "coding",
    "prompt": "### Instruction:\nWrite a Python function that reverses a linked list.\n\n### Response:\n",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### Python (OpenAI-compatible)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.completions.create(
    model="coding",          # ← name you gave the LoRA
    prompt="### Instruction:\nWrite a Python function that checks for a palindrome.\n\n### Response:\n",
    max_tokens=256,
    temperature=0.7,
)
print(response.choices[0].text)
```

---

## 5. What This Demonstrates

- **Dynamic adapter loading** – multiple specialized models share one base model in GPU memory
- **Continuous batching** – vLLM’s core performance feature
- **Multi-tenant serving** – different users/domains can hit different LoRAs without loading separate full models
- Clean separation between post-training (producing adapters) and inference engineering (serving them efficiently)

This directly matches the original project learning goal:

> Managing dynamic adapter loading (`vllm serve` with `--enable-lora`) to serve multiple fine-tuned variants of a model off a single base model instance.

---

## 6. Ready Script

See `scripts/serve_vllm.sh` for a ready-to-run wrapper.

---

## 7. Hardware Notes

| Environment | Feasible? | Notes |
|-------------|-----------|-------|
| Free Colab T4 | Borderline | May work with careful settings / shorter context |
| Colab Pro / A100 / L4 | Yes | Comfortable |
| Local NVIDIA ≥ 16 GB | Yes | Ideal |
| CPU only | No | Far too slow for 8B |

When you next have a GPU, just run the command in section 2 and you will have touched the full vLLM multi-LoRA path.
