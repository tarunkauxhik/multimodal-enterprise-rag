"""Separate in-memory rank-bm25 index.

Built from the chunk payloads already stored in Qdrant, so it can never drift
from the dense index and needs no second persistence format. Rebuild it after
ingestion or once at app start.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Bm25Index:
    payloads: list[dict]  # aligned with bm25 corpus order
    bm25: BM25Okapi | None  # None when there is nothing indexed


def build_bm25(payloads: Iterable[dict]) -> Bm25Index:
    # ponytail: full rebuild over all chunks; fine for V1 corpus sizes, add incremental/persisted index if startup gets slow
    payloads = list(payloads)
    bm25 = BM25Okapi([tokenize(p["text"]) for p in payloads]) if payloads else None
    return Bm25Index(payloads, bm25)
