"""Synthesize the gold eval set with Bulbul TTS (D008). Offline build step.

For each authored note in data/eval_scripts/*.jsonl: Bulbul v3 renders the
script (speaker rotated per language for variety), then ffmpeg WhatsApp-ifies
the audio — mono 16 kHz, mild pink noise, Opus in .ogg — into
data/eval_audio/<note_id>.ogg. Idempotent: existing outputs are skipped.

Usage:
  uv run --group synth python scripts/synthesize_eval_data.py [--language hi] [--limit 5] [--dry-run]

Needs SARVAM_API_KEY (see .env.example). Before the first full run, verify
Sarvam API ToS on synthetic-data reuse (D008). Validation after synthesis:
per-language listen-through of a sample + ASR round-trip against the script.
"""

import argparse
import base64
import os
import subprocess
import sys

from voice_digester.eval_data import EvalNote, load_eval_notes
from voice_digester.paths import eval_audio_dir

BULBUL_MODEL = "bulbul:v3"
LANGUAGE_CODES = {"hi": "hi-IN", "hi-en": "hi-IN", "ta": "ta-IN", "bn": "bn-IN"}
# Rotated per note (note index modulo len) so senders don't all share one voice.
SPEAKERS = {
    "hi": ["ritu", "amit", "pooja", "rahul"],
    "hi-en": ["kavya", "rohan", "simran", "dev"],
    "ta": ["kavitha", "mani", "shruti", "gokul"],
    "bn": ["ishita", "soham", "tanya", "kabir"],
}


def whatsappify(wav_bytes: bytes, out_path) -> None:
    """Mono 16 kHz + mild pink noise + Opus — approximates a WhatsApp voice note."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", "pipe:0",
        "-f", "lavfi", "-i", "anoisesrc=colour=pink:amplitude=0.012:sample_rate=16000",
        "-filter_complex",
        "[0:a]aresample=16000,pan=mono|c0=c0[v];[v][1:a]amix=inputs=2:duration=first[out]",
        "-map", "[out]", "-c:a", "libopus", "-b:a", "24k", "-vbr", "on",
        str(out_path),
    ]
    subprocess.run(cmd, input=wav_bytes, check=True)


def synthesize(note: EvalNote, speaker: str, client) -> bytes:
    response = client.text_to_speech.convert(
        text=note.script,
        model=BULBUL_MODEL,
        target_language_code=LANGUAGE_CODES[note.language],
        speaker=speaker,
        speech_sample_rate=16000,
    )
    return base64.b64decode("".join(response.audios))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=sorted(LANGUAGE_CODES), help="only this language")
    parser.add_argument("--limit", type=int, help="stop after N new files")
    parser.add_argument("--dry-run", action="store_true", help="list what would be synthesized")
    args = parser.parse_args()

    notes = load_eval_notes()
    if args.language:
        notes = [n for n in notes if n.language == args.language]
    plan = []
    for i, note in enumerate(notes):
        out_path = eval_audio_dir() / f"{note.note_id}.ogg"
        if out_path.exists():
            continue
        plan.append((note, SPEAKERS[note.language][i % len(SPEAKERS[note.language])], out_path))
    if args.limit:
        plan = plan[: args.limit]

    if args.dry_run:
        for note, speaker, out_path in plan:
            print(f"{note.note_id}  {LANGUAGE_CODES[note.language]}  speaker={speaker}  -> {out_path.name}")
        print(f"{len(plan)} note(s) to synthesize ({len(notes) - len(plan)} already done or filtered).")
        return 0

    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        print("SARVAM_API_KEY is not set (see .env.example).", file=sys.stderr)
        return 1
    from sarvamai import SarvamAI

    client = SarvamAI(api_subscription_key=api_key)
    eval_audio_dir().mkdir(parents=True, exist_ok=True)
    for note, speaker, out_path in plan:
        whatsappify(synthesize(note, speaker, client), out_path)
        print(f"wrote {out_path.name} (speaker={speaker})")
    print(f"done: {len(plan)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
