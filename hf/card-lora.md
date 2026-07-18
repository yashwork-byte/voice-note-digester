---
license: gemma
base_model: google/gemma-3-4b-it
library_name: peft
language: [hi, bn, ta, en]
tags: [lora, voice-notes, structured-output, action-items, translation, summarization, indic]
---

# gemma-3-4b-it-voice-digest (LoRA adapter)

LoRA adapter that turns **Gemma-3-4B-it** into a WhatsApp-voice-note digester
for Indic languages: given a raw STT transcript (Hindi, Hinglish, Tamil,
Bengali), it emits schema-exact JSON with an English **summary**,
**translation**, and **action items** carrying a `confirmed | tentative`
confidence tag — and returns an **empty task list** for purely conversational
notes instead of inventing tasks.

Part of the [voice-note-digester](https://github.com/yashwork-byte/voice-note-digester)
project. A ready-to-run Q4_K_M GGUF is at
[yashwork-byte/gemma-3-4b-it-voice-digest-GGUF](https://huggingface.co/yashwork-byte/gemma-3-4b-it-voice-digest-GGUF).

## Prompt format (must match exactly — the model was trained on it)

System prompt: see `system_prompt.txt` in this repo. User turn:

```
{instruction}

From: {sender}
{transcript}
```

Three trained instruction variants select the output format:

| Instruction | Output JSON |
|---|---|
| `Digest this voice note transcript.` | `{summary, translation, action_items}` |
| `Digest this voice note transcript. Skip the translation.` | `{summary, action_items}` (fast path) |
| `Translate this voice note transcript into English.` | `{translation}` |

`From:` matters: the sender is app-known metadata fed into the model so
summaries can name the sender without guessing (training labels are grounded
in it).

## Training

- 938 generated examples (844/94 split) across hi / hi-en / ta / bn —
  template × slot-fill generation with labels constructed from the same slot
  fills as the text (exact by construction). Inputs noised to STT shape
  (punctuation stripped). Dedicated slices for role attribution
  (sender-did-it vs listener-task), due-field discipline, confidence policy,
  implicit and question phrasings, and task-vocabulary pure chat.
- LoRA r=16 α=32 on all attention+MLP projections of the language model,
  bf16, 3 epochs, loss masked to the assistant JSON tokens.

## Evaluation (held-out, hand-authored, zero template overlap with training)

68 gold notes (17 per language) run **end-to-end**: synthetic WhatsApp-style
audio → IndicConformer-600M STT → this model (quantized Q4_K_M via llama.cpp
+ JSON grammar):

| Metric | Value |
|---|---|
| Action-item F1 (end-to-end) | **0.76** (bn 0.80 · hi 0.79 · ta 0.76 · hi-en 0.69) |
| Hallucinated-task rate on 20 pure-chat notes | 0.10 |
| Confidence-tag accuracy (matched items) | 0.90–1.00 |
| Translation chrF (end-to-end / clean text) | ~56 / ~64 |
| Schema-parse failures | 0 / 68 |

By phrasing style (end-to-end F1): imperative 0.86 · question 0.67 ·
colloquial 0.60 · implicit 0.25 (n=4, high variance).

## Limitations

- English proper nouns ("Odyssey") can arrive garbled from upstream Indic STT
  and be mis-normalized.
- Trained on 4 languages/registers of family-casual voice notes; other
  domains untested.
- Gemma derivative: use is subject to the
  [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
