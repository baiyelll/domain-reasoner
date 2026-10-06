# Connecting the Trained Adapter to Local `src/agent`

Your local structure:

```text
domain-reasoner/
├── src/
│   └── agent/
│       ├── __init__.py
│       ├── loop.py          ← main agent loop
│       ├── schema.py        ← message / state schemas
│       └── tools.py         ← tool definitions
├── tasks/
│   └── task_suite.py
└── ...
```

The trained adapter lives on Google Drive:

```text
adapters/swe-coding-agent-v1/     ← final 2-epoch adapter
```

Below is the recommended way to connect them.

---

## 1. Recommended Architecture

```text
┌─────────────────────┐
│   Agent Loop        │  (src/agent/loop.py)
│   - plan            │
│   - call tools      │
│   - observe         │
└─────────┬───────────┘
          │  prompts / messages
          ▼
┌─────────────────────┐
│   LLM Backend       │  ← new thin wrapper
│   (load adapter)    │
└─────────┬───────────┘
          │
          ▼
   Unsloth / PEFT model
   (base + LoRA adapter)
```

Keep the agent logic (`loop.py`, `tools.py`) separate from the model loading code.

---

## 2. Create a Model Backend

Add a new file, for example:

**`src/agent/llm.py`**

```python
# src/agent/llm.py
from __future__ import annotations
from typing import List, Dict, Optional
import torch

class CodingAgentLLM:
    """
    Thin wrapper around the fine-tuned Unsloth / PEFT adapter.
    """

    def __init__(
        self,
        adapter_path: str = "path/to/swe-coding-agent-v1",
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
    ):
        from unsloth import FastLanguageModel

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model)
        self.max_seq_length = max_seq_length

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        messages: list of {"role": "system"|"user"|"assistant", "content": "..."}
        Returns the assistant's reply as a string.
        """
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
            )

        # Remove the prompt tokens
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()
```

---

## 3. Wire It into `loop.py`

Typical pattern inside your agent loop:

```python
# inside src/agent/loop.py (conceptual)

from .llm import CodingAgentLLM
from .tools import available_tools   # your existing tools
from .schema import AgentState, Message

class Agent:
    def __init__(self, adapter_path: str):
        self.llm = CodingAgentLLM(adapter_path=adapter_path)
        self.tools = available_tools
        # ... other init

    def run(self, user_task: str, max_steps: int = 10) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert software engineering coding agent. "
                    "You can reason step by step and use tools when needed. "
                    "When you want to use a tool, output a structured tool call."
                ),
            },
            {"role": "user", "content": user_task},
        ]

        for step in range(max_steps):
            reply = self.llm.generate(messages)

            # 1. Try to parse tool calls from reply (your existing logic)
            tool_call = self.parse_tool_call(reply)   # implement according to your schema

            if tool_call is None:
                # Final answer
                return reply

            # 2. Execute tool
            observation = self.execute_tool(tool_call)

            # 3. Append to conversation
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"Tool result:\n{observation}"})

        return "Max steps reached."
```

You will need to adapt `parse_tool_call` and `execute_tool` to match whatever format you already defined in `schema.py` and `tools.py`.

---

## 4. How to Obtain the Adapter Locally

The adapter currently lives on Google Drive. Options:

**Option A – Download once**
```bash
# On your Mac (with rclone or Google Drive desktop)
# Download the folder:
# domain-reasoner/adapters/swe-coding-agent-v1/
```

Then point to it:
```python
adapter_path = "/Users/baielabdykerimov/.../swe-coding-agent-v1"
```

**Option B – Keep on Drive and mount** (if you run the agent in Colab / cloud)

**Option C – Later push to Hugging Face Hub** (private repo) and load by name.

---

## 5. Minimal Standalone Test (without full agent)

Before integrating the whole loop, test the backend alone:

```python
from src.agent.llm import CodingAgentLLM

llm = CodingAgentLLM(adapter_path="path/to/swe-coding-agent-v1")

messages = [
    {"role": "system", "content": "You are an expert coding agent."},
    {"role": "user", "content": "Write a Python function that reverses a linked list."},
]

print(llm.generate(messages))
```

---

## 6. Hardware Notes for Local Use

| Machine | Recommendation |
|---------|----------------|
| Apple Silicon (M1/M2/M3) | Possible with `load_in_4bit=False` + MPS, or use llama.cpp / MLX conversion later |
| NVIDIA GPU (local) | Ideal – same 4-bit loading as Colab |
| CPU only | Too slow for interactive 8B; not recommended |

If you only have a MacBook without NVIDIA GPU, consider:
- Keeping inference in Colab / Kaggle, or
- Converting the adapter to GGUF and using `llama-cpp-python`.

---

## 7. Suggested Integration Order

1. Download `swe-coding-agent-v1` from Drive to your machine (or keep path accessible).
2. Create `src/agent/llm.py` with the wrapper above.
3. Write a tiny script that only calls `llm.generate(...)` and prints the result.
4. Once that works, inject the LLM into `loop.py`.
5. Connect your existing tools (`tools.py`) via the parse → execute → observe cycle.

---

## 8. vLLM Alternative (for later)

When you move to a proper GPU server you can replace the Unsloth backend with vLLM:

```bash
vllm serve unsloth/Llama-3.1-8B-Instruct \
  --enable-lora \
  --lora-modules coding=./swe-coding-agent-v1
```

Then change `CodingAgentLLM` to call the OpenAI-compatible endpoint instead of loading the model in-process. This is the multi-tenant / continuous-batching path described in the original project goal.
