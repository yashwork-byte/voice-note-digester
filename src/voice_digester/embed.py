"""Content/query embeddings (D017).

all-MiniLM-L6-v2: 22M params, 384-dim, ONNX-able for phone deployment.
Summaries and tasks are embedded in English (the digest's target language),
so an English embedder suffices; paraphrase-multilingual-MiniLM-L12-v2 is the
same-dim upgrade path if Indic-language queries need first-class support.
"""

from functools import lru_cache

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _model().encode(texts, normalize_embeddings=True)]
