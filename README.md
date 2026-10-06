# Domain-Specific Reasoner  
### SFT → Agentic Trajectory SFT → vLLM-Ready Coding Agent

Fine-tuning Llama-3.1-8B into a **software-engineering coding agent** using parameter-efficient methods, then upgrading it with trajectory-based agentic SFT (tool use + environment feedback). Designed for efficient multi-tenant serving via vLLM + dynamic LoRA.


> **Status:** research prototype. Training and agent evaluation were executed; live vLLM serving was designed and scripted but **not yet run** (see [Limitations](#limitations)).

---

## Highlights

- **Full post-training pipeline on 14.6 GB of VRAM.** 4-bit QLoRA, 0.52% trainable parameters (41.9M / 8.07B), fp16, gradient checkpointing, resumable across Colab session resets.
- **Closed-loop agent training.** A subprocess-based execution harness provides ground-truth pass/fail signal; only successful rollouts are kept for SFT (rejection-sampling-style trajectory filtering).
- **Protocol failure was the first bottleneck.** The static-SFT model solved 2/28 rollouts under a strict JSON tool protocol. A 30-example format warmup raised that to 25/28.
- **Serving-ready artifacts.** Standard PEFT LoRA (r=16) adapters, loadable with Unsloth, Transformers + PEFT, or `vllm serve --enable-lora`.

---

## Method

```
Llama-3.1-8B-Instruct (4-bit)
        │
        ▼  Stage 1: static SFT (4,000 examples, 2 epochs, 1,000 steps)
  swe-coding-agent-v1
        │
        ▼  Stage 2a: format warmup (30 protocol-perfect trajectories, 4 epochs)
  swe-coding-agent-v1-format-warmup
        │
        ▼  Stage 2b: trajectory SFT (75 examples, 3 epochs, 30 steps)
  swe-coding-agent-v2-agentic
```

Each stage continues from the previous adapter.

### Stage 1: static domain SFT

| | |
|---|---|
| Base model | `unsloth/Llama-3.1-8B-Instruct` (4-bit NF4) |
| Adapter | LoRA r=16, α=16, dropout 0, all attention + MLP projections (`q,k,v,o,gate,up,down`) |
| Data | 4,000 ChatML examples sampled from Evol-Instruct-Code, Magicoder-Evol-Instruct, CodeFeedback |
| Schedule | 1,000 steps (2 epochs), effective batch 8 (2 × 4 accumulation), max length 2048 |
| Final train loss | ≈ 0.63 |
| Engineering | Unsloth + TRL `SFTTrainer`, checkpoints every 200 steps, resumed from step 600 after a runtime reset |

### Stage 2: agentic trajectory SFT

**Environment.** Seven self-contained bug-fix tasks (`is_palindrome`, `factorial`, `rate_limiter`, `longest_consecutive`, `dedupe_preserve_order`, `binary_search`, `merge_sorted`), each a buggy function or class plus assert-based tests. Offline, no pytest dependency.

**Harness.** Candidate code and tests run in an isolated subprocess with a timeout, so infinite loops (e.g. a broken binary search) are caught. The agent observes pass/fail plus truncated stdout/stderr.

**Tool protocol.** One JSON object per turn, nothing else:

```json
{"tool": "view_code"}
{"tool": "write_code", "code": "<full new source>"}
{"tool": "run_tests"}
{"tool": "finish"}
```

Rule enforced via the system prompt: always `run_tests` after `write_code` before `finish`.

**Pipeline.**

1. Roll out the policy on every task (k=4; one at T=0.3, three at T=0.8, ≤ 6 agent steps in the generation setting).
2. **Format warmup:** 30 hand-constructed, protocol-perfect trajectories (4 epochs, 16 steps) to lock the output format.
3. Re-roll with the warmed-up policy, keep successful trajectories.
4. **Trajectory SFT v2:** 75 examples, 3 epochs, 30 steps, LR 8e-5, cosine schedule, `packing=False`.

---

## Results

### Agent rollouts on the 7-task suite

| Policy | Rollouts passed | Notes |
|---|---|---|
| `v1` (static SFT only) | 2 / 28 (7%) | Frequently violates the JSON-only protocol (trailing prose, repeated `write_code`, step-budget exhaustion) |
| `v1` + format warmup | 25 / 28 (89%) | k=4 per task |
| `v2` (trajectory SFT) | 14 / 14 (100%) | k=2 per task |

**Read these numbers carefully.**

- The jump from 7% to 89% is most plausibly explained by **protocol adherence**, not new debugging ability: the failing `v1` traces show correct code being written but not emitted in the required format. We did not run a controlled ablation to confirm this.
- The suite is tiny and the trajectories used for `v2` were generated **on the same 7 tasks**, so the `v2` figure measures fit to the training tasks. It is evidence that the loop works end to end, not evidence of generalization.
- The evaluation cell re-declares the task definitions in a simplified form and uses a slightly different harness than generation, so the three rows are not strictly apples to apples.

### HumanEval slice

An evaluation harness for a 20-problem HumanEval slice (fixed seed 42, agent protocol with `run_tests` feedback) is included in the notebook. **A pass@1 result has not been recorded yet**, so none is reported here.

### Static SFT sanity checks

Qualitative generations from `v1` (palindrome check, `NoneType` debugging, an in-memory rate limiter, O(n) longest consecutive sequence) were coherent. The rate-limiter sample contained a logic error, which is consistent with the low rollout pass rate above. These are spot checks, not benchmarks.

---

## Quick start

### Inference

```python
from unsloth import FastLanguageModel
from peft import PeftModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
)
model = PeftModel.from_pretrained(model, "adapters/swe-coding-agent-v2-agentic")
FastLanguageModel.for_inference(model)
```

### Serving with vLLM (untested)

```bash
vllm serve unsloth/Llama-3.1-8B-Instruct \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 16 \
  --lora-modules \
      coding=./adapters/swe-coding-agent-v1 \
      agent=./adapters/swe-coding-agent-v2-agentic \
  --dtype float16 \
  --max-model-len 4096
```

Clients select an adapter via the OpenAI-compatible `model` field (`"coding"` or `"agent"`). Full instructions: [`SERVE.md`](SERVE.md).

Two caveats before you run this: adapters were trained against the **4-bit** base but this command serves the **fp16** base, so small output differences are possible; and fp16 weights for an 8B model need a GPU with roughly 24 GB (a T4 will not fit them without quantized serving).

---

## Reproducing

All experiments live in a Colab notebooks (`notebooks/domain_reasoner.ipynb`), organized as:

1. Dataset construction (ChatML, mixed sources, 4k examples)
2. Static QLoRA SFT with checkpoint/resume
3. Task suite and execution harness
4. Trajectory generation, format warmup, trajectory SFT
5. Agent and HumanEval-slice evaluation

Environment used: Colab Tesla T4 (14.6 GB), Unsloth 2026.9, Transformers 5.5.0, TRL ≤ 0.24, PEFT, bitsandbytes. The T4 has no bf16 support, so training runs in fp16.

---

## Repository layout

```text
domain-reasoner/
├── README.md
├── PROJECT_SUMMARY.md     # detailed technical log
├── INTEGRATION.md         # wiring an adapter into agent code
├── SERVE.md               # vLLM multi-LoRA serving guide
├── notebooks/
│   └── domain_reasoner.ipynb
├── scripts/
│   ├── train_sft.py
│   ├── eval_adapter.py
│   └── serve_vllm.sh
├── data/
│   └── README.md          # dataset description (no data committed)
└── requirements.txt
```

Adapter weights and the training JSONL are stored on Google Drive and are not part of this repository.

---

## Limitations

- **Small agent dataset.** 7 tasks, 75 SFT examples. This is a proof of mechanism. Scaling the task count (20–30+) and rollouts per task (k ≥ 8) is the obvious next lever.
- **No held-out agent evaluation.** The reported `v2` result is on training tasks. A held-out task split is needed before any generalization claim.
- **No ablations.** Warmup vs. no warmup, and trajectory SFT vs. warmup alone, are not separated.
- **HumanEval result missing.** The harness exists; the number does not.
- **Serving not validated.** The multi-LoRA vLLM path is documented and scripted but has not been executed on a GPU, and no throughput or latency measurements exist.
- **Format warmup data is synthetic.** The 30 warmup trajectories are templated, not model-generated.

---

## Roadmap

- [ ] Record HumanEval-20 pass@1 for base, `v1`, and `v2`
- [ ] Held-out bug-fix tasks and a warmup/no-warmup ablation
- [ ] Expand the task suite and rollout count; add a second filtering signal (e.g. step efficiency)
- [ ] Run and benchmark `vllm serve --enable-lora` with concurrent adapters
- [ ] Preference or RL stage (e.g. rejection-sampling fine-tuning → GRPO) using harness reward

---

## Acknowledgements and licenses

Built with [Unsloth](https://github.com/unslothai/unsloth), Hugging Face TRL / PEFT / Transformers, bitsandbytes, and vLLM. Base model: Llama 3.1 Community License. Instruction data: Evol-Instruct-Code, Magicoder-Evol-Instruct, CodeFeedback-Filtered-Instruction (each under its own license). This repository contains code and documentation only.

---

**Author:** Baiel Abdykerimov · October 2026
