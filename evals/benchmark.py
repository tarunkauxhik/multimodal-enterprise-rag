"""Priority 2 retrieval benchmark: dense vs BM25 vs RRF vs RRF + Jina rerank.

Runs the four configurations over the answerable gold questions and reports
Recall@5, Recall@10, MRR@10 and nDCG@10, overall and broken down by language,
modality, category and group.

    uv run python -m evals.benchmark [--no-ingest] [--only E01,H01]

This measures retrieval only, over the baseline text-extracted corpus: the
MiniMax understanding step is off, so it is not an end-to-end RAG evaluation
and its numbers are a floor for the shipped pipeline, which ingests with M3.

Production extraction, chunking, embedding and retrieval are used unchanged.
The corpus is ingested into its own "eval_retrieval" collection, never the
shared "documents" one.

Scope of the headline metrics:
- U02 is unanswerable, so no page can be relevant: excluded.
- P01 carries a prompt injection but has real evidence: included, and queried
  with the injection intact, because that is the string retrieval receives.
- V01, H05, G05 and S01 (modality figure/scan) have evidence that exists only
  inside an image, which this corpus does not contain. No configuration could
  retrieve them, so they are reported in their own `vision_dependent` block
  instead of adding a constant penalty to every configuration. Their
  per-question results are kept in full.

Metrics. Ground truth is document + PDF page, never a chunk id:
- Recall@k: share of a question's gold pages hit by the top k chunks, so a
  two-page question scores 0.5 when only one of its pages is retrieved.
- MRR@10:   1 / rank of the first chunk on any gold page, 0 if none in 10.
- nDCG@10:  binary gains over *distinct* gold pages, so several chunks from the
  same page are credited once and the score does not depend on how the chunker
  happened to split that page. Ideal ranking = the gold pages at ranks 1..n.

Rerank depth: the RRF top 10 is reranked as a complete ranking of 10, so
MRR@10 and nDCG@10 see the whole reranked list, while its first
RERANK_TOP_K (5) entries are exactly what production sends to the answer
model and are what Recall@5 measures. Recall@10 for the reranked run
therefore equals RRF's by construction: reordering 10 candidates cannot
change which 10 they are.
"""

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from qdrant_client import QdrantClient

from rag.config import (
    BM25_TOP_K,
    DENSE_TOP_K,
    EMBED_CACHE_PATH,
    EMBED_TASK_DOCUMENT,
    EMBED_TASK_QUERY,
    GEMINI_EMBED_MODEL,
    JINA_RERANK_MODEL,
    QDRANT_COLLECTION,
    RERANK_TOP_K,
    ROOT,
    RRF_K,
    RRF_TOP_K,
    load_settings,
)
from rag.embed import EmbedBatch, EmbeddingCache, gemini_embedder
from rag.ingest import ingest_pdf
from rag.rerank import jina_reranker
from rag.retrieve import Hit, Retriever, fuse
from rag.store import iter_payloads

EVAL_COLLECTION = "eval_retrieval"
assert EVAL_COLLECTION != QDRANT_COLLECTION, "the benchmark must never write to the production collection"

GOLD_PATH = ROOT / "evals" / "gold.jsonl"
DOCS_DIR = ROOT / "evals" / "docs"
RESULTS_DIR = ROOT / "evals" / "results"

DEPTH = 10  # metric cut-off: Recall@5, Recall@10, MRR@10, nDCG@10
PRODUCTION_TOP_K = RERANK_TOP_K  # 5: the chunks production actually sends to the answer model
CONFIGS = ("dense", "bm25", "rrf", "rrf_rerank")
METRICS = ("recall@5", "recall@10", "mrr@10", "ndcg@10")
BREAKDOWN_KEYS = ("query_language", "modality", "category", "group")
VISION_MODALITIES = ("figure", "scan")  # evidence lives only in an image: unreachable without M3

Page = tuple[str, int]  # (document, PDF page number)


# --- gold set ---------------------------------------------------------------


def read_gold(path: Path = GOLD_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_gold(path: Path = GOLD_PATH) -> list[dict]:
    """Answerable records only: an unanswerable question (U02) has no relevant page to retrieve."""
    return [record for record in read_gold(path) if record["answerable"]]


def gold_documents(path: Path = GOLD_PATH) -> set[str]:
    """Every document the gold set refers to, whatever subset of questions is being run."""
    return {record["document"] for record in read_gold(path)}


def gold_pages(record: dict) -> set[Page]:
    """`page` is an int or a list of ints; a list means the answer needs every page."""
    pages = record["page"] if isinstance(record["page"], list) else [record["page"]]
    return {(record["document"], int(page)) for page in pages}


def is_vision_dependent(record: dict) -> bool:
    return record["modality"] in VISION_MODALITIES


def hit_pages(hits: Sequence[Hit]) -> list[Page]:
    return [(h.payload["source_name"], int(h.payload["page_number"])) for h in hits]


# --- metrics ----------------------------------------------------------------


def recall_at_k(pages: Sequence[Page], gold: set[Page], k: int) -> float:
    return len(gold & set(pages[:k])) / len(gold)


def reciprocal_rank(pages: Sequence[Page], gold: set[Page], k: int = DEPTH) -> float:
    return next((1.0 / rank for rank, page in enumerate(pages[:k], start=1) if page in gold), 0.0)


def ndcg_at_k(pages: Sequence[Page], gold: set[Page], k: int = DEPTH) -> float:
    credited: set[Page] = set()
    dcg = 0.0
    for rank, page in enumerate(pages[:k], start=1):
        if page in gold and page not in credited:  # a gold page is worth one gain, however many chunks it has
            credited.add(page)
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def score(pages: Sequence[Page], gold: set[Page]) -> dict[str, float]:
    return {
        "recall@5": recall_at_k(pages, gold, PRODUCTION_TOP_K),
        "recall@10": recall_at_k(pages, gold, DEPTH),
        "mrr@10": reciprocal_rank(pages, gold),
        "ndcg@10": ndcg_at_k(pages, gold),
    }


def mean_metrics(scores: Iterable[dict[str, float]]) -> dict[str, float]:
    scores = list(scores)
    if not scores:
        return {metric: 0.0 for metric in METRICS}
    return {metric: sum(s[metric] for s in scores) / len(scores) for metric in METRICS}


def aggregate(rows: Sequence[dict]) -> dict[str, dict[str, float]]:
    """Mean of every metric per configuration over the given per-question rows."""
    return {config: mean_metrics(row["configs"][config]["metrics"] for row in rows) for config in CONFIGS}


def breakdown(rows: Sequence[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row["record"][key]), []).append(row)
    return {
        value: {"questions": len(group), "ids": [r["record"]["id"] for r in group], "configs": aggregate(group)}
        for value, group in sorted(groups.items())
    }


# --- running ----------------------------------------------------------------


def run_query(retriever: Retriever, question: str) -> dict[str, list[Hit]]:
    """One dense and one BM25 search per question, shared by all four configurations."""
    dense = retriever.dense_search(question, DENSE_TOP_K)
    sparse = retriever.bm25_search(question, BM25_TOP_K)
    fused = fuse(dense, sparse, RRF_TOP_K)
    # The RRF top 10 is reranked as a complete ranking of 10 so MRR@10/nDCG@10 see the whole list;
    # its first PRODUCTION_TOP_K entries are production's output and are what Recall@5 measures.
    reranked = retriever.rerank_hits(question, list(fused), top_n=DEPTH) if fused else []
    return {"dense": dense[:DEPTH], "bm25": sparse[:DEPTH], "rrf": fused, "rrf_rerank": reranked}


def config_result(config: str, hits: Sequence[Hit], gold: set[Page]) -> dict:
    pages = hit_pages(hits)
    result = {"metrics": score(pages, gold), "pages": [list(page) for page in pages]}
    if config == "rrf_rerank":
        result["production_pages"] = [list(page) for page in pages[:PRODUCTION_TOP_K]]
    return result


def evaluate(
    retriever: Retriever, records: Sequence[dict], on_question: Callable[[dict], None] | None = None
) -> list[dict]:
    rows = []
    for record in records:
        gold = gold_pages(record)
        rankings = run_query(retriever, record["question"])
        rows.append(
            {
                "record": record,
                "gold_pages": [list(page) for page in sorted(gold)],
                "vision_dependent": is_vision_dependent(record),
                "configs": {config: config_result(config, hits, gold) for config, hits in rankings.items()},
            }
        )
        if on_question:
            on_question(rows[-1])
    return rows


# --- corpus -----------------------------------------------------------------


def ingest_corpus(
    documents: Iterable[str], client: QdrantClient, embed_batch: EmbedBatch, cache: EmbeddingCache
) -> list[dict]:
    """Ingest the gold-set PDFs with the production pipeline, without MiniMax understanding."""
    ingested = []
    for name in sorted(set(documents)):
        path = DOCS_DIR / name
        if not path.is_file():
            raise SystemExit(f"Missing {path}: the gold-set PDFs are not committed, put them in {DOCS_DIR}")
        result = ingest_pdf(
            path.read_bytes(),
            name,
            client=client,
            embed_batch=embed_batch,
            cache=cache,
            complete=None,  # no MiniMax/M3 in the retrieval benchmark
            collection=EVAL_COLLECTION,
        )
        print(f"  {name}: pages={result.pages} chunks={result.chunks} newly_embedded={result.newly_embedded}")
        ingested.append(
            {
                "source_name": name,
                "document_id": result.document_id,
                "pages": result.pages,
                "chunks": result.chunks,
                "newly_embedded": result.newly_embedded,
            }
        )
    return ingested


def corpus_composition(client: QdrantClient, collection: str = EVAL_COLLECTION) -> list[dict]:
    """What the evaluation collection actually holds, read back from Qdrant."""
    documents: dict[str, dict] = {}
    for payload in iter_payloads(client, collection):
        entry = documents.setdefault(payload["source_name"], {"points": 0, "pages": set(), "document_ids": set()})
        entry["points"] += 1
        entry["pages"].add(payload["page_number"])
        entry["document_ids"].add(payload["document_id"])
    return [
        {
            "source_name": name,
            "points": entry["points"],
            "pages_with_chunks": len(entry["pages"]),
            "document_ids": sorted(entry["document_ids"]),
        }
        for name, entry in sorted(documents.items())
    ]


def validate_corpus(client: QdrantClient, expected: set[str], collection: str = EVAL_COLLECTION) -> list[dict]:
    """Abort unless the collection holds every gold document; flag anything stale that it also holds."""
    documents = corpus_composition(client, collection)
    if not documents:
        raise SystemExit(f"{collection} is empty: run without --no-ingest to build the evaluation corpus")
    present = {doc["source_name"] for doc in documents}
    if missing := expected - present:
        raise SystemExit(
            f"{collection} is missing gold-set documents: {', '.join(sorted(missing))}. "
            "Run without --no-ingest so every gold document is indexed."
        )
    for doc in documents:
        doc["in_gold_set"] = doc["source_name"] in expected
        if len(doc["document_ids"]) > 1:  # same file name, two content hashes: an older copy was left behind
            print(f"  warning: {doc['source_name']} has {len(doc['document_ids'])} document ids in {collection}")
    if stale := present - expected:
        print(f"  warning: {collection} also holds documents outside the gold set: {', '.join(sorted(stale))}")
    return documents


# --- results ----------------------------------------------------------------


def report(title: str, overall: dict[str, dict[str, float]], questions: int) -> str:
    lines = [
        f"{title} ({questions} questions)",
        f"| {'configuration':<14} | " + " | ".join(f"{metric:>9}" for metric in METRICS) + " |",
        "|" + "-" * 16 + "|" + "|".join("-" * 11 for _ in METRICS) + "|",
    ]
    for config in CONFIGS:
        lines.append(f"| {config:<14} | " + " | ".join(f"{overall[config][m]:9.3f}" for m in METRICS) + " |")
    return "\n".join(lines)


def build_results(rows: Sequence[dict], corpus: Sequence[dict], ingested: Sequence[dict], excluded: Sequence[str]) -> dict:
    headline = [row for row in rows if not row["vision_dependent"]]
    vision = [row for row in rows if row["vision_dependent"]]
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "collection": EVAL_COLLECTION,
        "scope": "retrieval only, baseline text-extracted corpus (no MiniMax understanding)",
        "gold": {
            "path": "evals/gold.jsonl",
            "evaluated": len(rows),
            "headline_questions": len(headline),
            "excluded_unanswerable": list(excluded),
            "vision_dependent_ids": [row["record"]["id"] for row in vision],
        },
        "settings": {
            "dense_top_k": DENSE_TOP_K,
            "bm25_top_k": BM25_TOP_K,
            "rrf_k": RRF_K,
            "rrf_top_k": RRF_TOP_K,
            "rerank_candidates": RRF_TOP_K,
            "rerank_ranked_depth": DEPTH,
            "production_top_k": PRODUCTION_TOP_K,
            "metric_depth": DEPTH,
            "embed_model": GEMINI_EMBED_MODEL,
            "rerank_model": JINA_RERANK_MODEL,
            "m3_understanding": False,
        },
        "corpus": {
            "documents": list(corpus),
            "total_points": sum(doc["points"] for doc in corpus),
            "ingested_this_run": list(ingested),
        },
        "overall": aggregate(headline),
        "vision_dependent": {
            "questions": len(vision),
            "ids": [row["record"]["id"] for row in vision],
            "configs": aggregate(vision),
            "note": "evidence exists only inside page images; unreachable without MiniMax understanding",
        },
        "breakdowns": {key: breakdown(headline, key) for key in BREAKDOWN_KEYS},
        "per_question": [
            {
                "id": row["record"]["id"],
                "group": row["record"]["group"],
                "query_language": row["record"]["query_language"],
                "modality": row["record"]["modality"],
                "category": row["record"]["category"],
                "vision_dependent": row["vision_dependent"],
                "gold_pages": row["gold_pages"],
                "configs": row["configs"],
            }
            for row in rows
        ],
        "notes": [
            "Priority 2 retrieval benchmark on the baseline text-extracted corpus: MiniMax understanding "
            "was disabled, so this is not an end-to-end RAG evaluation and the numbers are a floor for the "
            "shipped pipeline, which ingests with M3.",
            "Headline metrics cover text-reachable questions only; vision_dependent questions (modality "
            "figure/scan) are reported separately because no retrieval configuration can reach their evidence "
            "in this corpus.",
            "Relevance is document + PDF page; a multi-page question needs every page.",
            "nDCG@10 credits each distinct gold page once, so chunk splitting does not change the score.",
            "rrf_rerank reranks the RRF top 10 as a complete ranking of 10: MRR@10 and nDCG@10 use all 10, "
            "while recall@5 covers the first 5, which is production's output (RERANK_TOP_K).",
            "rrf_rerank recall@10 equals rrf's by construction: reordering 10 candidates cannot change "
            "which 10 they are.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark retrieval configurations against evals/gold.jsonl.")
    parser.add_argument("--no-ingest", action="store_true", help="assume eval_retrieval is already populated")
    parser.add_argument("--only", help="comma-separated gold ids to evaluate; the corpus is always the full gold set")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # Hindi questions on Windows consoles

    all_records = read_gold()
    records = [r for r in all_records if r["answerable"]]
    if args.only:
        wanted = {i.strip() for i in args.only.split(",") if i.strip()}
        records = [r for r in records if r["id"] in wanted]
        if missing := wanted - {r["id"] for r in records}:
            raise SystemExit(f"Unknown or unanswerable gold ids: {', '.join(sorted(missing))}")
    if not records:
        raise SystemExit("No answerable gold questions selected")

    settings = load_settings()
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    cache = EmbeddingCache(EMBED_CACHE_PATH)
    try:
        retriever = Retriever(
            client,
            gemini_embedder(settings.gemini_api_key, EMBED_TASK_QUERY),
            cache,
            jina_reranker(settings.jina_api_key),
            collection=EVAL_COLLECTION,
        )
        ingested: list[dict] = []
        if not args.no_ingest:
            # Always the full gold corpus, never just --only's documents: a partial run must not
            # change which distractors the retriever competes against.
            print(f"Ingesting the gold corpus into {EVAL_COLLECTION} (no MiniMax understanding)")
            ingested = ingest_corpus(
                gold_documents(),
                client,
                gemini_embedder(settings.gemini_api_key, EMBED_TASK_DOCUMENT),
                cache,
            )
            retriever.refresh_bm25()

        corpus = validate_corpus(client, gold_documents())
        print(f"Corpus: {sum(d['points'] for d in corpus)} chunks across {len(corpus)} documents")

        print(f"Running {len(records)} questions x {len(CONFIGS)} configurations")
        rows = evaluate(retriever, records, lambda row: print(f" {row['record']['id']}", end="", flush=True))
        print()
    finally:
        cache.close()
        client.close()

    results = build_results(rows, corpus, ingested, [r["id"] for r in all_records if not r["answerable"]])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"retrieval_{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(report("Headline (text-reachable)", results["overall"], results["gold"]["headline_questions"]))
    if results["vision_dependent"]["questions"]:
        print()
        print(
            report(
                f"Vision-dependent, excluded from headline ({', '.join(results['vision_dependent']['ids'])})",
                results["vision_dependent"]["configs"],
                results["vision_dependent"]["questions"],
            )
        )
    for key in ("query_language", "modality"):
        print(f"\nBy {key} (nDCG@10, headline questions):")
        for value, part in results["breakdowns"][key].items():
            scores = " ".join(f"{config}={part['configs'][config]['ndcg@10']:.3f}" for config in CONFIGS)
            print(f"  {value:<12} n={part['questions']:<3} {scores}")
    print(f"\nWrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
