"""Spoken language ID (D013): VoxLingua107 ECAPA (~21 MB) restricted to a
3-way choice — code-mixed Hinglish lands on "hi", which is its STT code anyway."""

from functools import lru_cache

LID_MODEL = "speechbrain/lang-id-voxlingua107-ecapa"
CANDIDATES = {"hi", "ta", "bn"}


@lru_cache(maxsize=1)
def _classifier():
    from speechbrain.inference import EncoderClassifier

    clf = EncoderClassifier.from_hparams(source=LID_MODEL)
    ind2lab = clf.hparams.label_encoder.ind2lab
    index = {lab.split(":")[0].strip(): ind for ind, lab in ind2lab.items()
             if lab.split(":")[0].strip() in CANDIDATES}
    return clf, index


def detect(wav) -> str:
    """`wav` is a mono 16 kHz float32 tensor [1, T] (see stt.decode_audio)."""
    clf, index = _classifier()
    log_probs = clf.classify_batch(wav)[0][0]
    return max(index, key=lambda code: log_probs[index[code]])
