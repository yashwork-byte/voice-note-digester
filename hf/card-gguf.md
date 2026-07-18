---
license: gemma
base_model: google/gemma-3-4b-it
language: [hi, bn, ta, en]
tags: [gguf, llama.cpp, voice-notes, structured-output, action-items, indic]
---

# gemma-3-4b-it-voice-digest-GGUF

**Q4_K_M GGUF (~2.4 GB)** of the merged
[gemma-3-4b-it-voice-digest LoRA](https://huggingface.co/yashwork-byte/gemma-3-4b-it-voice-digest-lora):
a WhatsApp-voice-note digester for Hindi / Hinglish / Tamil / Bengali that
emits schema-exact JSON (English summary, translation, `confirmed|tentative`
action items — empty list for pure chat). Runs on llama.cpp anywhere,
including phones; ~25 s per note on an 8 GB M2 Air, all offline.

See the adapter card for the **required prompt format** (system prompt,
`From: {sender}` line, and the three instruction variants that select
full / no-translation / translation-only output), training data, and the
full evaluation. Headline: end-to-end action-item F1 0.76, hallucination
0.10 on pure-chat notes, 0 schema-parse failures, measured through this
exact quantized file with a JSON grammar.

## Quick start (llama-cpp-python)

```python
from llama_cpp import Llama
llm = Llama.from_pretrained("yashwork-byte/gemma-3-4b-it-voice-digest-GGUF",
                            "gemma-3-4b-it-ft-Q4_K_M.gguf", n_ctx=2048, n_gpu_layers=-1)
out = llm.create_chat_completion(
    messages=[{"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user", "content": f"Digest this voice note transcript.\n\nFrom: Papa\n{transcript}"}],
    response_format={"type": "json_object"}, temperature=0.1)
```

Gemma derivative — subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
