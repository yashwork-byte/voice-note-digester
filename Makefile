config ?= sarvam-translate-4b.yaml
audio ?=
q ?=

# Synthesize the gold eval set with Bulbul TTS (offline build step, needs SARVAM_API_KEY).
# Pass args="--dry-run" / args="--language ta --limit 5" to scope the run.
synth-data:
	uv run --group synth python scripts/synthesize_eval_data.py $(args)

transcribe:
	uv run --group stt python -m voice_digester.stt $(audio) $(lang)

# D008 validation: transcribe all synthesized eval audio on Modal, WER vs scripts.
asr-round-trip:
	uv run --group eval modal run -m voice_digester.asr_round_trip

digest:
	uv run --group digest python -m voice_digester.digest "$(text)"

index:
	uv run python -m voice_digester.vector_store

search:
	uv run python -m voice_digester.search "$(q)"

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
