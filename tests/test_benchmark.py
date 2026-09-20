"""Benchmark metric, scoping and corpus-validation tests (offline: in-memory Qdrant, fake services)."""

import json

import pytest
from qdrant_client import QdrantClient

from evals.benchmark import (
    CONFIGS,
    DEPTH,
    EVAL_COLLECTION,
    PRODUCTION_TOP_K,
    aggregate,
    breakdown,
    build_results,
    corpus_composition,
    evaluate,
    gold_documents,
    gold_pages,
    hit_pages,
    is_vision_dependent,
    load_gold,
    ndcg_at_k,
    read_gold,
    recall_at_k,
    reciprocal_rank,
    run_query,
    score,
    validate_corpus,
)
from rag.chunk import Chunk
from rag.config import EMBED_TASK_DOCUMENT, QDRANT_COLLECTION, RERANK_TOP_K
from rag.embed import EmbeddingCache, embed_texts
from rag.retrieve import Hit, Retriever, fuse
from rag.store import ensure_collection, replace_document
from tests.conftest import hash_embed

DOC = "report.pdf"
GOLD = {(DOC, 5)}
MULTI = {(DOC, 5), (DOC, 6)}
VISION_IDS = {"V01", "H05", "G05", "S01"}


def pages(*numbers, document=DOC):
    return [(document, n) for n in numbers]


def record(question_id="Z", page=4, modality="text", question="paid annual leave rule 3"):
    return {
        "id": question_id,
        "question": question,
        "document": DOC,
        "page": page,
        "answerable": True,
        "query_language": "en",
        "modality": modality,
        "category": "text",
        "group": "test",
    }


# --- gold set ---------------------------------------------------------------


def test_only_answerable_questions_are_evaluated():
    answerable = load_gold()
    ids = [r["id"] for r in answerable]
    assert "U02" not in ids  # unanswerable: no page can be "relevant"
    assert "P01" in ids  # prompt injection, but it has legitimate evidence
    assert len(answerable) == len(read_gold()) - 1
    assert all(r["answerable"] for r in answerable)


def test_gold_pages_accepts_an_int_or_a_list():
    assert gold_pages({"document": DOC, "page": 5}) == GOLD
    assert gold_pages({"document": DOC, "page": [5, 6]}) == MULTI
    multi = [r for r in load_gold() if isinstance(r["page"], list)]
    assert {r["id"] for r in multi} == {"E04", "M05"} and all(len(gold_pages(r)) == 2 for r in multi)


def test_hit_pages_reads_document_and_page_from_the_payload():
    hits = [Hit({"source_name": DOC, "page_number": 5}), Hit({"source_name": "other.pdf", "page_number": 2})]
    assert hit_pages(hits) == [(DOC, 5), ("other.pdf", 2)]


def test_vision_dependent_questions_are_identified_by_modality():
    assert {r["id"] for r in read_gold() if is_vision_dependent(r)} == VISION_IDS
    assert not is_vision_dependent(record(modality="table"))
    assert is_vision_dependent(record(modality="scan"))


def test_corpus_documents_cover_the_whole_gold_set_whatever_is_evaluated():
    """--only must not shrink the corpus: the distractors have to stay the same."""
    assert gold_documents() == {r["document"] for r in read_gold()}
    assert len(gold_documents()) == 3
    assert gold_documents() >= {r["document"] for r in load_gold() if r["id"] == "E01"}


# --- metrics ----------------------------------------------------------------


def test_recall_counts_gold_pages_not_chunks():
    # four chunks from the one gold page still count once, and give full recall
    assert recall_at_k(pages(5, 5, 5, 5), GOLD, 5) == 1.0
    assert recall_at_k(pages(1, 2, 3), GOLD, 5) == 0.0


def test_recall_gives_partial_credit_for_multi_page_questions():
    assert recall_at_k(pages(5, 1, 2), MULTI, 5) == 0.5
    assert recall_at_k(pages(5, 6), MULTI, 5) == 1.0


def test_recall_at_5_and_10_respect_the_cut_off():
    ranking = pages(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)  # gold page 5 sits at rank 5, page 6 at rank 6
    assert recall_at_k(ranking, MULTI, 5) == 0.5
    assert recall_at_k(ranking, MULTI, DEPTH) == 1.0


def test_reciprocal_rank_uses_the_first_relevant_chunk_and_is_zero_past_the_depth():
    assert reciprocal_rank(pages(5), GOLD) == 1.0
    assert reciprocal_rank(pages(1, 2, 5, 5), GOLD) == pytest.approx(1 / 3)
    assert reciprocal_rank(pages(11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 5), GOLD) == 0.0  # gold page at rank 11
    assert reciprocal_rank(pages(1, 2), GOLD) == 0.0


def test_ndcg_is_one_for_the_ideal_ranking_and_falls_with_rank():
    assert ndcg_at_k(pages(5), GOLD) == 1.0
    assert ndcg_at_k(pages(5, 6), MULTI) == 1.0
    assert ndcg_at_k(pages(6, 5), MULTI) == 1.0  # both gold pages on top: order between them is irrelevant
    assert ndcg_at_k(pages(1, 5), GOLD) == pytest.approx(1 / 1.5849625007211562)
    assert ndcg_at_k(pages(1, 2, 3), GOLD) == 0.0


def test_ndcg_credits_a_gold_page_once_so_chunk_splitting_cannot_inflate_it():
    assert ndcg_at_k(pages(5, 1, 2), GOLD) == ndcg_at_k(pages(5, 5, 5), GOLD) == 1.0
    # a second gold page found at rank 2 must beat the same page repeated
    assert ndcg_at_k(pages(5, 6), MULTI) > ndcg_at_k(pages(5, 5), MULTI)


def test_score_reports_all_four_metrics_with_recall_at_5_at_the_production_depth():
    assert PRODUCTION_TOP_K == RERANK_TOP_K == 5
    assert score(pages(1, 5, 6), MULTI) == {
        "recall@5": 1.0,
        "recall@10": 1.0,
        "mrr@10": pytest.approx(0.5),
        "ndcg@10": pytest.approx((1 / 1.5849625007211562 + 1 / 2.0) / (1 + 1 / 1.5849625007211562)),
    }
    # a gold page at rank 6 is outside production's top 5 but inside the metric depth
    at_six = score(pages(1, 2, 3, 4, 7, 5), GOLD)
    assert at_six["recall@5"] == 0.0 and at_six["recall@10"] == 1.0


# --- aggregation ------------------------------------------------------------


def row(question_id, language, value, modality="text"):
    return {
        "record": {
            "id": question_id,
            "query_language": language,
            "modality": modality,
            "category": "text",
            "group": "test",
        },
        "vision_dependent": modality in ("figure", "scan"),
        "gold_pages": [[DOC, 5]],
        "configs": {config: {"metrics": flat(value), "pages": [[DOC, 5]]} for config in CONFIGS},
    }


def flat(value):
    return {"recall@5": value, "recall@10": value, "mrr@10": value, "ndcg@10": value}


def test_aggregate_averages_each_metric_per_configuration():
    assert aggregate([row("A", "en", 1.0), row("B", "hi", 0.0)])["dense"] == flat(0.5)


def test_breakdown_splits_by_field_and_keeps_the_question_ids():
    rows = [row("A", "en", 1.0), row("B", "hi", 0.0), row("C", "hi", 1.0)]
    by_language = breakdown(rows, "query_language")
    assert by_language["en"] == {"questions": 1, "ids": ["A"], "configs": aggregate(rows[:1])}
    assert by_language["hi"]["questions"] == 2 and by_language["hi"]["ids"] == ["B", "C"]
    assert by_language["hi"]["configs"]["rrf"]["ndcg@10"] == 0.5


def test_vision_questions_are_split_out_of_the_headline_but_kept_per_question():
    rows = [row("A", "en", 1.0), row("B", "en", 1.0), row("V01", "en", 0.0, modality="figure")]
    results = build_results(rows, [{"source_name": DOC, "points": 3}], [], ["U02"])

    assert results["overall"]["dense"] == flat(1.0)  # the zero-scoring vision question does not drag it down
    assert results["gold"]["headline_questions"] == 2 and results["gold"]["evaluated"] == 3
    assert results["vision_dependent"]["ids"] == ["V01"]
    assert results["vision_dependent"]["configs"]["dense"] == flat(0.0)
    assert [q["id"] for q in results["per_question"]] == ["A", "B", "V01"]  # nothing deleted
    assert [q["vision_dependent"] for q in results["per_question"]] == [False, False, True]
    assert all("V01" not in part["ids"] for part in results["breakdowns"]["query_language"].values())


def test_results_record_scope_and_corpus_provenance():
    corpus = [{"source_name": DOC, "points": 42, "pages_with_chunks": 9, "document_ids": ["abc"], "in_gold_set": True}]
    results = build_results([row("A", "en", 1.0)], corpus, [{"source_name": DOC, "chunks": 42}], ["U02"])
    assert results["settings"]["m3_understanding"] is False
    assert "not an end-to-end RAG evaluation" in " ".join(results["notes"])
    assert results["scope"].startswith("retrieval only")
    assert results["corpus"]["total_points"] == 42
    assert results["corpus"]["documents"][0]["document_ids"] == ["abc"]
    assert results["corpus"]["ingested_this_run"][0]["chunks"] == 42


# --- corpus validation ------------------------------------------------------


def chunks_for(document, page_numbers, doc_id="d"):
    return [
        Chunk(f"{doc_id}-p{page}-{i}", doc_id, document, page, ("Report",), "text", f"paid annual leave rule {i}")
        for i, page in enumerate(page_numbers)
    ]


def load(client, cache, chunks, doc_id="d"):
    vectors, _ = embed_texts([c.text for c in chunks], EMBED_TASK_DOCUMENT, hash_embed, cache)
    replace_document(client, doc_id, chunks, vectors, EVAL_COLLECTION)


@pytest.fixture
def qdrant(tmp_path):
    client = QdrantClient(":memory:")
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    ensure_collection(client, EVAL_COLLECTION)
    yield client, cache
    cache.close()
    client.close()


def test_empty_corpus_aborts(qdrant):
    client, _ = qdrant
    with pytest.raises(SystemExit, match="empty"):
        validate_corpus(client, {DOC})


def test_missing_gold_document_aborts_and_names_it(qdrant):
    client, cache = qdrant
    load(client, cache, chunks_for(DOC, [1, 2]))
    with pytest.raises(SystemExit, match="missing.pdf"):
        validate_corpus(client, {DOC, "missing.pdf"})


def test_corpus_composition_records_points_pages_and_document_ids(qdrant):
    client, cache = qdrant
    load(client, cache, chunks_for(DOC, [1, 1, 2]))
    load(client, cache, chunks_for("other.pdf", [7], doc_id="e"), doc_id="e")
    composition = corpus_composition(client)
    assert [doc["source_name"] for doc in composition] == ["other.pdf", DOC]
    report = next(doc for doc in composition if doc["source_name"] == DOC)
    assert report["points"] == 3 and report["pages_with_chunks"] == 2 and report["document_ids"] == ["d"]


def test_stale_documents_are_flagged_but_do_not_abort(qdrant, capsys):
    client, cache = qdrant
    load(client, cache, chunks_for(DOC, [1]))
    load(client, cache, chunks_for("leftover.pdf", [1], doc_id="e"), doc_id="e")
    documents = validate_corpus(client, {DOC})
    assert "leftover.pdf" in capsys.readouterr().out
    assert {doc["source_name"]: doc["in_gold_set"] for doc in documents} == {DOC: True, "leftover.pdf": False}


# --- configurations against a real (in-memory) index ------------------------


class ReversingReranker:
    """Scores by reverse position, so the reranked order differs from the RRF order."""

    def __init__(self):
        self.calls = []

    def __call__(self, query, documents, top_n):
        self.calls.append((query, list(documents), top_n))
        return [(i, float(i)) for i in range(len(documents))][::-1][:top_n]


@pytest.fixture
def retriever(qdrant):
    client, cache = qdrant
    load(client, cache, chunks_for(DOC, list(range(1, 26))))
    reranker = ReversingReranker()
    return Retriever(client, hash_embed, cache, reranker, collection=EVAL_COLLECTION), reranker


def test_run_query_produces_the_four_configurations_at_the_metric_depth(retriever):
    retriever, reranker = retriever
    rankings = run_query(retriever, "paid annual leave rule 3")

    assert set(rankings) == set(CONFIGS)
    assert all(len(hits) <= DEPTH for hits in rankings.values())
    assert len(rankings["dense"]) == len(rankings["bm25"]) == len(rankings["rrf"]) == DEPTH

    (query, documents, top_n), = reranker.calls  # the RRF top 10, ranked in full
    assert query == "paid annual leave rule 3" and len(documents) == DEPTH and top_n == DEPTH
    assert len(rankings["rrf_rerank"]) == DEPTH
    assert [h.payload["text"] for h in rankings["rrf_rerank"]] == documents[::-1]
    assert [h.payload["text"] for h in rankings["rrf"]] == documents  # the RRF ranking is not mutated


def test_reranked_top_5_is_the_production_retrieval_output(retriever):
    retriever, _ = retriever
    question = "paid annual leave rule 3"
    fused = fuse(retriever.dense_search(question), retriever.bm25_search(question))
    production = retriever.rerank_hits(question, list(fused))  # production default top_n = RERANK_TOP_K

    benchmark = run_query(retriever, question)["rrf_rerank"]
    assert len(production) == PRODUCTION_TOP_K
    assert [h.chunk_id for h in benchmark[:PRODUCTION_TOP_K]] == [h.chunk_id for h in production]


def test_production_pages_are_recorded_for_the_reranked_run(retriever):
    retriever, _ = retriever
    (result,) = evaluate(retriever, [record()])
    reranked = result["configs"]["rrf_rerank"]
    assert reranked["production_pages"] == reranked["pages"][:PRODUCTION_TOP_K]
    assert len(reranked["pages"]) == DEPTH
    assert "production_pages" not in result["configs"]["rrf"]


def test_dense_and_bm25_are_searched_once_and_shared_by_every_configuration(retriever):
    retriever, _ = retriever
    calls = []
    original = retriever.dense_search
    retriever.dense_search = lambda *a, **kw: (calls.append(a), original(*a, **kw))[1]
    run_query(retriever, "paid annual leave")
    assert len(calls) == 1


def test_reranking_cannot_change_recall_at_10_but_can_change_the_top_5(retriever):
    retriever, _ = retriever
    (result,) = evaluate(retriever, [record()])
    rrf, reranked = result["configs"]["rrf"]["metrics"], result["configs"]["rrf_rerank"]["metrics"]

    assert rrf["recall@10"] == reranked["recall@10"]  # same candidate set, only reordered
    assert set(map(tuple, result["configs"]["rrf"]["pages"])) == set(map(tuple, reranked and result["configs"]["rrf_rerank"]["pages"]))
    assert result["gold_pages"] == [[DOC, 4]]
    assert rrf["mrr@10"] != reranked["mrr@10"]  # the reversing reranker moves the relevant chunk


def test_benchmark_never_touches_the_production_collection(retriever):
    retriever, _ = retriever
    run_query(retriever, "paid annual leave")
    assert retriever.collection == EVAL_COLLECTION != QDRANT_COLLECTION
    assert not retriever.client.collection_exists(QDRANT_COLLECTION)


def test_results_are_json_serialisable(retriever):
    retriever, _ = retriever
    rows = evaluate(retriever, [record(page=[4, 5]), record("V01", page=6, modality="figure")])
    results = build_results(rows, corpus_composition(retriever.client), [], ["U02"])

    loaded = json.loads(json.dumps(results, ensure_ascii=False))
    assert loaded["collection"] == EVAL_COLLECTION
    assert loaded["gold"]["excluded_unanswerable"] == ["U02"]
    assert loaded["gold"]["vision_dependent_ids"] == ["V01"]
    assert set(loaded["overall"]) == set(CONFIGS)
    assert loaded["per_question"][0]["gold_pages"] == [[DOC, 4], [DOC, 5]]
    assert loaded["corpus"]["total_points"] == 25
