"""Separate in-memory rank-bm25 index.

Built from the chunk payloads already stored in Qdrant, so it can never drift
from the dense index and needs no second persistence format. Rebuild it after
ingestion or once at app start.
"""

import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Lowercased runs of letters, combining marks and digits, in any script.

    Unicode categories are used instead of regex \\w, which splits Devanagari
    words at vowel signs ("कर्मचारियों" -> "कर", "मच", ...).
    """
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch if unicodedata.category(ch)[0] in "LMN" else " " for ch in text).split()


@dataclass
class Bm25Index:
    payloads: list[dict]  # aligned with bm25 corpus order
    bm25: BM25Okapi | None  # None when there is nothing indexed

    def search(self, query: str, top_k: int) -> list[tuple[dict, float]]:
        """Chunks sharing at least one query token, best BM25 score first."""
        tokens = tokenize(query)
        if self.bm25 is None or not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        # Match on shared tokens, not score > 0: rank_bm25 can give real matches a score <= 0 in small corpora.
        matching = [i for i, freqs in enumerate(self.bm25.doc_freqs) if any(t in freqs for t in tokens)]
        matching.sort(key=lambda i: scores[i], reverse=True)
        return [(self.payloads[i], float(scores[i])) for i in matching[:top_k]]


def build_bm25(payloads: Iterable[dict]) -> Bm25Index:
    # ponytail: full rebuild over all chunks; fine for V1 corpus sizes, add incremental/persisted index if startup gets slow
    payloads = list(payloads)
    bm25 = BM25Okapi([tokenize(p["text"]) for p in payloads]) if payloads else None
    return Bm25Index(payloads, bm25)
