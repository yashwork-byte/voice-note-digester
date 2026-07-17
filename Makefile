config ?= gemma-3-4b-it.yaml
ft_config ?= ft-gemma-3-4b-lora.yaml
checkpoint ?=
audio ?=
q ?=

# Synthesize the gold eval set with Bulbul TTS (offline build step, needs SARVAM_API_KEY).
# Pass args="--dry-run" / args="--language ta --limit 5" to scope the run.
synth-data:
	uv run --group synth python scripts/synthesize_eval_data.py $(args)

# Generate the fine-tuning dataset (deterministic; D018).
train-data:
	uv run python scripts/generate_training_data.py

# LoRA SFT on Modal (D019). Smoke-test first: make fine-tune (with max_steps in the yaml).
fine-tune:
	uv run --group eval modal run --detach -m voice_digester.fine_tune --config-file-name $(ft_config)

# Merge adapter + quantize to the fixed volume path: make export-gguf checkpoint=<run_name>
export-gguf:
	uv run --group eval modal run --detach -m voice_digester.export_gguf --checkpoint $(checkpoint)

transcribe:
	uv run --group stt python -m voice_digester.stt $(audio) $(lang)

# D008 validation: transcribe all synthesized eval audio on Modal, WER vs scripts.
asr-round-trip:
	uv run --group eval modal run -m voice_digester.asr_round_trip

digest:
	uv run --group digest python -m voice_digester.digest "$(text)"

# Build data/notes.db from saved digest-eval predictions (no model runs).
ingest:
	uv run --group store python -m voice_digester.ingest

search:
	uv run --group store python -m voice_digester.search "$(q)"

# D004/D015 validation, on Modal. source=script isolates digest quality;
# source=transcript is end-to-end (includes STT errors).
source ?= transcript
evaluate:
	uv run --group eval modal run --detach -m voice_digester.evaluate --config-file-name $(config) --source $(source)

test:
	uv run --with pytest pytest -q

lint:
	uv run ruff check --fix .

format:
	uv run ruff format .
