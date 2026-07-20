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

# Pull the fine-tuned GGUF + int8 STT from the Modal volume for local inference.
fetch-model:
	uv run --group eval modal volume get --force voice-digester gguf/gemma-3-4b-it-ft-Q4_K_M.gguf data/models/
	uv run --group eval modal volume get --force voice-digester stt-int8 data/models/indic-conformer-int8

# Run the live demo at http://localhost:8000 — ALL inference is local (D023).
demo:
	uv run --env-file .env --group demo uvicorn demo.app:app --port 8000

# Deploy the GPU-backed inference API to Modal (portfolio demo; D024).
deploy-web:
	uv run --group eval modal deploy -m voice_digester.web

# Assemble the Vercel frontend bundle pointing at the Modal URL:
#   make vercel-bundle url=https://<you>--suno-web-fastapi-app.modal.run
vercel-bundle:
	rm -rf demo/dist && cp -r demo/static demo/dist
	printf 'window.SUNO_API = "%s";\n' "$(url)" > demo/dist/config.js
	@echo "bundle ready in demo/dist — deploy with:  cd demo/dist && vercel --prod"

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
