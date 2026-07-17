"""Single place for project path construction (house style)."""

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return project_root() / "data"


def eval_scripts_dir() -> Path:
    """Gold eval scripts (JSON, authored with labels — D008)."""
    return data_dir() / "eval_scripts"


def eval_audio_dir() -> Path:
    """Bulbul-synthesized, WhatsApp-ified audio for the gold scripts (D008)."""
    return data_dir() / "eval_audio"


def processed_dir() -> Path:
    return data_dir() / "processed"


def db_path() -> Path:
    """The local sqlite-vec database (D006)."""
    return data_dir() / "notes.db"
