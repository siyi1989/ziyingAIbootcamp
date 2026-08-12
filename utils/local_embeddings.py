import hashlib
import re
from typing import List

from langchain_core.embeddings import Embeddings


class LocalHashEmbeddings(Embeddings):
    """A lightweight deterministic embedding fallback that does not require
    external API calls. It is intentionally simple and works well for local
    proof-of-concept retrieval when cloud embeddings are unavailable or rate-limited."""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

def _normalize(self, text: str) -> str:
    text = text.lower()

    # Keep original customer code forms
    original = text

    # Add split version of alphanumeric codes
    expanded = re.sub(
        r"([a-z]+)(\d+)",
        r"\1 \2",
        text
    )

    combined = f"{original} {expanded}"

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        combined
    ).strip()

    def _embed_text(self, text: str) -> List[float]:
        tokens = self._normalize(text).split()
        if not tokens:
            return [0.0] * self.dimension

        vector = [0.0] * self.dimension
        for token in tokens:
            idx = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self.dimension
            vector[idx] += 1.0

        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0:
            return [0.0] * self.dimension
        return [value / norm for value in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)
