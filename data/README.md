# Dataset

## File
- `swe_coding_agent_seed.jsonl` (stored on Google Drive, not in git)

## Format (ChatML)
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert software engineering coding agent. You write clean, correct, and well-explained code. When fixing bugs or implementing features, reason step by step and provide the final solution clearly."
    },
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ]
}
```

## Sources
- nickrosh/Evol-Instruct-Code-80k-v1
- ise-uiuc/Magicoder-Evol-Instruct-110K
- m-a-p/CodeFeedback-Filtered-Instruction

## Size
4,000 cleaned and formatted examples used for training.
