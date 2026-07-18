# Suno — voice-note digester

> Speak a WhatsApp-style voice note in Hindi, Hinglish, Tamil, or Bengali —
> get an English summary, translation, and confidence-tagged action items,
> all searchable, all on-device.

---

##  What is Suno?

Family voice notes pile up. Suno turns each one into structured, searchable
data — locally, with no audio or text leaving the machine.

Instead of:
- replaying a 40-second note to find one errand
- forgetting which note mentioned the venue advance

You can:
> record (or share) a note → get summary + tasks → ask "what do I need to do
> for the wedding?"

The hard part isn't summarization — it's restraint. Most family notes are
pure chat, and inventing a task nobody asked for is worse than missing one.
The whole project is built and measured around that failure mode.

---

##  What it does

- Detects the spoken language automatically (hi / ta / bn; Hinglish → hi)
- Transcribes with IndicConformer-600M, int8-quantized ONNX (0.6 GB)
- Digests with a LoRA fine-tune of Gemma-3-4B-it (Q4 GGUF, llama.cpp,
  JSON-grammar constrained) — one structured call per note
- Tags every action item `confirmed` ("book by Friday") or `tentative`
  ("maybe we should think about it") — and returns an empty list for chat
- Generates the translation lazily, only when you open it
- Indexes summaries and action items as separate entity types in sqlite-vec
- Answers natural-language queries against both, results labeled note/task

---

##  System Architecture

    Voice note → Language ID → STT (int8 ONNX) → Digest — one structured call
                 (ECAPA, 21MB)  (IndicConformer)  (Gemma-3-4B-it + LoRA,
                                                   Q4 GGUF, llama.cpp + grammar)
                                                        ↓
                                     summary + action items  [translation on demand]
                                                        ↓
                                  sqlite-vec — two entity types: notes / tasks
                                                        ↓
                       query → embedded once → BOTH tables → results labeled note/task

    Modal (training + evaluation only — inference never leaves the device):

        taxonomy → train data → LoRA → merge → Q4 GGUF ──▶ one-file swap into Digest
                                                  │
                        68-note gold eval ◀───────┘

---

##  The taxonomy

The eval design adapts the instruction taxonomy from the
[Liquid AI cookbook home-assistant example](https://github.com/Liquid4All/cookbook/tree/main/examples/home-assistant)
to speech. Every gold note is classified along:

- **Language** — hi · hi-en (code-mixed) · ta · bn
- **Phrasing** — imperative ("book the venue") · colloquial ("tu le lena
  yaar…") · question ("could you pick it up?") · implicit ("no one's home to
  receive the delivery…")
- **Confidence** — confirmed vs tentative
- **Boundary** — no-action-items: pure chat that must yield `[]`, including
  adversarial notes full of task vocabulary (sender-already-did-it reports,
  "I'll handle X" notes, care advice, gossip about other people's bookings)

Gold set: 68 hand-authored notes (17/language), labels authored *with* the
scripts, voiced by TTS, degraded to WhatsApp audio, never trained on.
Training set: 938 examples generated from the same taxonomy (template ×
slot-fill), so every eval slice has a training slice.

Every eval run reports per-slice — which is what turned scores into
diagnoses: prompt engineering only *moved* failures between phrasing styles
(imperative up, colloquial down — a seesaw); the base model silently dropped
tentative and question-form asks; and a training-data leak (summaries naming
senders the model couldn't know) was localized in one query session.

---

##  Workflow

    author gold notes ─▶ TTS + whatsapp-ify ─▶ ASR round-trip (validates audio)
    (taxonomy-complete)                                  │
                                                         ▼
    generate train set ─▶ LoRA (Modal) ─▶ GGUF ─▶ eval on the untouched 68:
    (same taxonomy)                               clean-text AND end-to-end,
            ▲                                     sliced by language × phrasing
            │                                                │
            └────────── fix data / prompt / format ◀─────────┘
                        (never eval-blind)

Two invariants:
- Evals run the exact production path — quantized GGUF, JSON grammar, real
  STT transcripts — so quantization and speech errors are inside the number.
- Training code imports its instruction strings *from* the runtime code, so
  train/serve skew is structurally impossible.

---

##  Results

Held-out, end-to-end (audio → STT → quantized model):

| Metric | Base Gemma-3-4B | Fine-tuned |
|---|---|---|
| Action-item F1 | ~0.55 | **0.76** |
| Hallucinated tasks on pure-chat notes | 0.05–0.20 (prompt seesaw) | **0.10** |
| Confidence-tag accuracy | 0.17–1.0 (scrambled) | **0.90–1.00** |
| Translation chrF | ~47 | **~56** |
| Schema-parse failures | 0 | **0** |

~25 s per note warm on an 8 GB M2 Air, fully local.

Models:
[LoRA](https://huggingface.co/yashwork-byte/gemma-3-4b-it-voice-digest-lora) ·
[GGUF](https://huggingface.co/yashwork-byte/gemma-3-4b-it-voice-digest-GGUF) ·
[int8 STT](https://huggingface.co/yashwork-byte/indic-conformer-600m-int8-onnx)

---

##  Setup

### 1. Clone the repo

    git clone https://github.com/yashwork-byte/voice-note-digester
    cd voice-note-digester

### 2. Install dependencies ([uv](https://docs.astral.sh/uv/) + ffmpeg)

    uv sync --group demo

### 3. Environment

Create `.env` in the root (IndicConformer and Gemma are gated on HF):

    HF_TOKEN=hf_...

### 4. Models

Place under `data/models/` (from the HF repos above):

    data/models/gemma-3-4b-it-ft-Q4_K_M.gguf
    data/models/indic-conformer-int8/

---

##  Run

    make demo        # http://localhost:8000 — wait for "warmup complete" in the log

Tap the mic, speak, tap again. First note after startup is the slowest.

Everything else is a make target: `train-data`, `fine-tune`, `export-gguf`,
`evaluate` (Modal) · `synth-data` (rebuild eval audio) · `ingest`, `search`
(local store) · `test`.

---

##  Key Design Ideas

- Build the eval before the pipeline; never promote a model the untouched
  gold set hasn't scored
- One model, three trained output formats (full / no-translation /
  translation-only) — grammar-forcing an untrained format degraded quality;
  training the formats fixed it
- The model only outputs what it can know: sender is fed *in* (`From: Papa`),
  metadata is attached by the app, never generated
- Summaries and tasks are separate searchable entities — no query router to
  guess wrong
- "No action items" is a first-class output, with adversarial eval notes to
  keep it honest

---

##  Limitations

- English proper nouns can garble at the STT stage ("Odyssey" → "ओडीसी")
  and get mis-normalized downstream
- Four languages, family-casual register; other domains untested
- Implicit-task eval slice is small (n=4) — its numbers are noisy
- ~25 s per note is the 4B floor on 8 GB hardware

---

##  Future Work

- Due-date filtering in search ("anything due Friday" needs date logic, not
  embeddings)
- Character-level STT noise in training data (targets the last hallucination
  residual)
- Gemma-3-1B variant through the same harness (~4× faster, quality cost
  measurable)
- Per-sender language priors

---

##  Credits & Licenses

Digest model is a Gemma derivative
([Gemma Terms of Use](https://ai.google.dev/gemma/terms)). STT derives from
[AI4Bharat IndicConformer](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
(MIT). Eval audio synthesized with Sarvam Bulbul TTS. Taxonomy design
inspired by the Liquid AI cookbook. Eval data referenced: AI4Bharat
IndicVoices / BhasaAnuvaad (CC BY 4.0).
