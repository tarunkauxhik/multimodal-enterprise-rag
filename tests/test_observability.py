"""Ingestion observability: the IngestReport filled by ingest_pdf (offline: in-memory Qdrant, fakes)."""

import json
import math
import os
import sqlite3
import subprocess
import sys
import threading
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from qdrant_client import QdrantClient

from rag import embed as embed_module
from rag import ingest as ingest_module
from rag import session
from rag import store as store_module
from rag.chunk import Chunk
from rag.config import EMBED_BATCH_SIZE, EMBED_DIM, EMBED_TASK_DOCUMENT, QDRANT_COLLECTION
from rag.embed import EmbeddingCache, EmbedStats, embed_texts, gemini_embedder
from rag.extract import extract_pdf
from rag.ingest import STAGES, IngestReport, StoreStats, format_report, ingest_pdf
from rag.store import ensure_collection, iter_payloads, replace_document
from rag.understand import PagePlan, RoutingStats, UnderstandingCache, plan_page, routing_stats
from tests.conftest import RecordingModel, build_text_pdf, hash_embed
from tests.test_understand import FakeM3

SECRET = "sk-test-secret"
SAMPLE_SENTENCE = "paragraph line 0 describing the company results"  # text inside sample_pdf


class CountingEmbedder:
    def __init__(self):
        self.batches = 0

    def __call__(self, texts):
        self.batches += 1
        return hash_embed(texts)


@pytest.fixture
def env(tmp_path):
    client = QdrantClient(":memory:")
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    understanding = UnderstandingCache(tmp_path / "u.sqlite")
    yield SimpleNamespace(client=client, cache=cache, understanding=understanding)
    cache.close()
    understanding.close()
    client.close()


def ingest(env, pdf, *, complete=None, embedder=None, report=None, name="sample.pdf"):
    return ingest_pdf(pdf, name, client=env.client, embed_batch=embedder or CountingEmbedder(), cache=env.cache,
                      complete=complete, understanding_cache=env.understanding, report=report)


# --- happy path ------------------------------------------------------------------


def test_happy_path_fills_identity_stage_and_timings(env, sample_pdf):
    result = ingest(env, sample_pdf, complete=FakeM3())
    report = result.report
    assert report.stage == "done" and report.error is None
    assert (report.source_name, report.document_id, report.pages) == ("sample.pdf", result.document_id, 2)
    assert report.empty_pages == []
    assert set(report.seconds) == {*STAGES, "total"} and all(v >= 0 for v in report.seconds.values())
    assert report.seconds["total"] + 0.01 >= sum(report.seconds[s] for s in STAGES)  # rounding slack


def test_existing_result_fields_are_unchanged(env, sample_pdf):
    result = ingest(env, sample_pdf, complete=FakeM3())
    assert result.chunks == sum(result.report.chunks_by_type.values())
    assert result.understood_pages == [1] and result.failed_pages == []
    assert result.newly_embedded == result.report.embedding.new


# --- routing -------------------------------------------------------------------


def test_routing_counts_match_plan_page(env, sample_pdf):
    routing = ingest(env, sample_pdf, complete=FakeM3()).report.routing
    plans = [plan_page(p) for p in extract_pdf(sample_pdf, "sample.pdf").pages]
    assert routing.m3_enabled
    assert routing.routed_pages == sum(p.needed for p in plans) == 1  # page 1 has the chart
    assert routing.normal_pages == 1 and routing.reasons == {"figure": 1} and routing.overlaps == {}
    assert routing.understood == 1 and routing.failed == [] and routing.from_cache == 0


def test_unique_routed_pages_are_counted_apart_from_reasons_and_overlaps():
    plans = [
        PagePlan(transcribe=True, reasons=("legacy_font",)),
        PagePlan(transcribe=True, figures=(2,), reasons=("legacy_font", "figure")),
        PagePlan(figures=(1,), reasons=("figure",)),
        PagePlan(transcribe=True, figures=(3,), reasons=("scanned", "figure")),
        PagePlan(),
    ]
    stats = routing_stats(plans, m3_enabled=True)
    assert stats.routed_pages == 4 and stats.normal_pages == 1  # 4 M3 calls, not 6 reason hits
    assert stats.reasons == {"legacy_font": 2, "figure": 3, "scanned": 1}
    assert stats.overlaps == {"legacy_font+figure": 1, "scanned+figure": 1}


def test_routing_is_reported_when_m3_is_disabled(env, sample_pdf):
    routing = ingest(env, sample_pdf, complete=None).report.routing
    assert not routing.m3_enabled
    assert routing.routed_pages == 1 and routing.reasons == {"figure": 1}  # what M3 would have been asked
    assert routing.understood == 0 and routing.failed == [] and routing.from_cache == 0


def test_m3_failures_are_recorded_as_page_numbers(env, sample_pdf):
    routing = ingest(env, sample_pdf, complete=RecordingModel(RuntimeError("M3 down"))).report.routing
    assert routing.understood == 0 and routing.failed == [1]


def test_m3_cache_hits_are_counted(env, sample_pdf):
    model = FakeM3()
    first = ingest(env, sample_pdf, complete=model).report.routing
    second = ingest(env, sample_pdf, complete=model).report.routing
    assert (first.understood, first.from_cache) == (1, 0)
    assert (second.understood, second.from_cache) == (1, 1)
    assert len(model.calls) == 1  # the second run never called M3


# --- chunks and embedding --------------------------------------------------------


def test_chunk_type_counts_add_up_to_the_chunk_total(env, sample_pdf):
    result = ingest(env, sample_pdf, complete=FakeM3())
    assert sum(result.report.chunks_by_type.values()) == result.chunks
    assert set(result.report.chunks_by_type) == {"text", "table", "figure"}  # the sample has all three


def test_embedding_counts_new_then_cached(env, sample_pdf):
    embedder = CountingEmbedder()
    first = ingest(env, sample_pdf, embedder=embedder).report.embedding
    assert first.texts >= first.unique > 0
    assert (first.new, first.cached) == (first.unique, 0)
    assert first.batches_attempted == first.batches_succeeded == embedder.batches == math.ceil(first.unique / EMBED_BATCH_SIZE)

    second = ingest(env, sample_pdf, embedder=embedder).report.embedding
    assert (second.new, second.cached) == (0, second.unique)
    assert second.batches_attempted == 0 and embedder.batches == first.batches_attempted


class FlakyModels:
    """Fake google-genai `client.models`: raises the scripted transport errors, then answers."""

    def __init__(self, failures):
        self.failures = list(failures)

    def embed_content(self, *, model, contents, config):
        if self.failures:
            raise self.failures.pop(0)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.5] * EMBED_DIM) for _ in contents])


def test_our_transport_retries_are_counted(monkeypatch, tmp_path):
    models = FlakyModels([httpx.ReadError("[SSL: BAD_RECORD_MAC]"), httpx.RemoteProtocolError("disconnected")])
    monkeypatch.setattr(embed_module.genai, "Client", lambda **kw: SimpleNamespace(models=models))
    stats, cache = EmbedStats(), EmbeddingCache(tmp_path / "e.sqlite")
    embed_texts(["a", "b"], EMBED_TASK_DOCUMENT, gemini_embedder("k", EMBED_TASK_DOCUMENT, sleep=lambda s: None), cache, stats=stats)
    cache.close()
    assert stats.transport_retries == 2 and stats.batches_succeeded == 1


def test_embedder_retries_outside_an_embed_texts_call_are_harmless(monkeypatch):
    monkeypatch.setattr(embed_module.genai, "Client", lambda **kw: SimpleNamespace(models=FlakyModels([httpx.ReadError("x")])))
    assert len(gemini_embedder("k", EMBED_TASK_DOCUMENT, sleep=lambda s: None)(["a"])) == 1  # no active stats


def test_transport_retries_are_isolated_between_concurrent_ingestions(monkeypatch, tmp_path):
    """The API shares one embedder across threads: each ingestion must count only its own retries."""
    lock, barrier = threading.Lock(), threading.Barrier(2, timeout=10)
    failures_left, started = {"a": 2, "b": 1}, set()

    class SharedModels:
        def embed_content(self, *, model, contents, config):
            who = contents[0].parts[0].text[0]
            with lock:
                first = who not in started
                started.add(who)
            if first:
                barrier.wait()  # both ingestions are inside embed_texts at the same time
            with lock:
                fail = failures_left[who] > 0
                failures_left[who] -= fail
            if fail:
                raise httpx.ReadError("connection dropped")
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.5] * EMBED_DIM) for _ in contents])

    monkeypatch.setattr(embed_module.genai, "Client", lambda **kw: SimpleNamespace(models=SharedModels()))
    shared = gemini_embedder("k", EMBED_TASK_DOCUMENT, sleep=lambda s: None)  # one embedder, as in build_services
    results, errors = {}, []

    def run(who):
        cache = EmbeddingCache(tmp_path / f"{who}.sqlite")
        try:
            stats = EmbedStats()
            embed_texts([f"{who} chunk {i}" for i in range(3)], EMBED_TASK_DOCUMENT, shared, cache, stats=stats)
            results[who] = stats
        except Exception as exc:  # surfaced below; an exception in a thread would otherwise vanish
            errors.append(exc)
        finally:
            cache.close()

    threads = [threading.Thread(target=run, args=(who,)) for who in "ab"]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert not errors
    assert results["a"].transport_retries == 2 and results["b"].transport_retries == 1  # a shared counter would say 3


# --- storage -------------------------------------------------------------------


def test_replace_document_returns_the_number_of_stale_points_deleted(env):
    ensure_collection(env.client)
    chunks = [Chunk(f"docA-p1-{i}", "docA", "a.pdf", 1, ("S",), "text", f"text {i}") for i in range(3)]
    vectors = [[1.0] * EMBED_DIM] * 3
    assert replace_document(env.client, "docA", chunks, vectors) == 0
    assert replace_document(env.client, "docA", chunks[:1], vectors[:1]) == 2
    assert replace_document(env.client, "docA", chunks[:1], vectors[:1]) == 0


def test_storage_counts_and_stale_deletion_on_reingest(env, sample_pdf, monkeypatch):
    first = ingest(env, sample_pdf, name="sample.pdf")
    other = ingest(env, build_text_pdf(["Another document entirely."]), name="other.pdf")
    storage = first.report.storage
    assert storage.upserted == storage.document_points == first.chunks and storage.stale_deleted == 0
    assert other.report.storage.collection_points == first.chunks + other.chunks

    real_chunker = ingest_module.chunk_document
    monkeypatch.setattr(ingest_module, "chunk_document", lambda doc: real_chunker(doc)[:-1])  # e.g. a chunker change
    again = ingest(env, sample_pdf, name="sample.pdf").report.storage
    assert again.stale_deleted == 1 and again.document_points == first.chunks - 1
    assert again.collection_points == first.chunks - 1 + other.chunks  # the other document is untouched


# --- failures ------------------------------------------------------------------


class FailingSecondBatch:
    def __init__(self, error):
        self.calls, self.error = 0, error

    def __call__(self, texts):
        self.calls += 1
        if self.calls == 2:
            raise self.error
        return hash_embed(texts)


def test_partial_embedding_failure_keeps_progress_and_re_raises_the_same_exception(env, sample_pdf, monkeypatch):
    monkeypatch.setattr(ingest_module, "embed_texts", partial(embed_texts, batch_size=1))  # one chunk per batch
    boom = RuntimeError("Gemini embedding failed: ClientError")
    report = IngestReport("sample.pdf")

    with pytest.raises(RuntimeError) as raised:
        ingest(env, sample_pdf, embedder=FailingSecondBatch(boom), report=report)

    assert raised.value is boom  # the exact original exception object
    assert report.stage == "embed" and report.error == "RuntimeError"
    e = report.embedding
    assert (e.batches_attempted, e.batches_succeeded, e.new) == (2, 1, 1)
    assert "embed" in report.seconds and "store" not in report.seconds and "total" in report.seconds
    assert report.storage == StoreStats() and not env.client.collection_exists(QDRANT_COLLECTION)

    retry = CountingEmbedder()
    resumed = ingest(env, sample_pdf, embedder=retry).report.embedding
    assert resumed.cached == 1 and resumed.new == resumed.unique - 1  # the saved batch is reused


def test_unreadable_pdf_still_raises_value_error_and_reports_the_extract_stage(env):
    report = IngestReport("bad.pdf")
    with pytest.raises(ValueError, match="Not a readable PDF"):
        ingest(env, b"not a pdf", report=report, name="bad.pdf")
    assert (report.stage, report.error, report.document_id) == ("extract", "ValueError", "")


def test_session_ingest_upload_fills_the_callers_report_on_failure(fake_services, sample_pdf):
    collection = session.start_session(fake_services.client)

    def quota_exceeded(texts):
        raise RuntimeError("quota exceeded")

    fake_services.embed_document = quota_exceeded
    report = IngestReport("sample.pdf")
    with pytest.raises(RuntimeError, match="quota exceeded"):
        session.ingest_upload(fake_services, collection, sample_pdf, "sample.pdf", report=report)
    assert report.stage == "embed" and report.error == "RuntimeError" and report.pages == 2


# --- privacy and serialisation ---------------------------------------------------


def test_reports_never_contain_document_text_or_secrets(env, sample_pdf, monkeypatch):
    monkeypatch.setattr(ingest_module, "embed_texts", partial(embed_texts, batch_size=1))
    failed = IngestReport("sample.pdf")
    leaky = RuntimeError(f"HTTP 401 key={SECRET} while embedding '{SAMPLE_SENTENCE}'")
    with pytest.raises(RuntimeError):  # before the successful run, which would cache every chunk
        ingest(env, sample_pdf, embedder=FailingSecondBatch(leaky), report=failed)
    ok = ingest(env, sample_pdf, complete=FakeM3()).report

    assert SAMPLE_SENTENCE in " ".join(b.text for p in extract_pdf(sample_pdf, "s").pages for b in p.blocks)
    for report in (ok, failed):
        dumped = json.dumps(report.to_dict()) + format_report(report)
        assert SAMPLE_SENTENCE not in dumped and "paragraph line" not in dumped
        assert SECRET not in dumped and "HTTP 401" not in dumped  # exception class only, never its message


def test_report_round_trips_through_json(env, sample_pdf):
    report = ingest(env, sample_pdf, complete=FakeM3()).report
    loaded = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))
    assert loaded["stage"] == "done" and loaded["document_id"] == report.document_id
    assert loaded["routing"]["reasons"] == {"figure": 1} and loaded["routing"]["understood"] == 1
    assert set(loaded["embedding"]) == {"texts", "unique", "cached", "new", "batches_attempted",
                                        "batches_succeeded", "transport_retries"}
    assert set(loaded["storage"]) == {"upserted", "stale_deleted", "document_points", "collection_points"}
    assert set(loaded["seconds"]) == {*STAGES, "total"}


# --- CLI -----------------------------------------------------------------------


def known_report(**overrides):
    fields = dict(
        source_name="mhi.pdf", document_id="ba5c6d183cb01260", pages=266, stage="done", empty_pages=[2, 4],
        routing=RoutingStats(m3_enabled=True, normal_pages=80, routed_pages=186,
                             reasons={"legacy_font": 130, "figure": 106, "scanned": 8, "table": 1},
                             overlaps={"legacy_font+figure": 60, "scanned+figure": 3},
                             understood=184, failed=[12, 47], from_cache=120),
        chunks_by_type={"figure": 44, "table": 180, "text": 812},
        embedding=EmbedStats(texts=1036, unique=1012, cached=180, new=832, batches_attempted=167,
                             batches_succeeded=167, transport_retries=2),
        storage=StoreStats(upserted=1036, stale_deleted=3, document_points=1036, collection_points=3114),
        seconds={"extract": 6.1, "understand": 28.0, "chunk": 0.4, "embed": 5.2, "store": 1.1, "total": 40.8},
    )
    return IngestReport(**(fields | overrides))


def test_format_report_is_concise_and_covers_every_stage():
    text = format_report(known_report())
    lines = text.splitlines()
    assert lines[0] == "mhi.pdf  id=ba5c6d183cb01260  266 pages  total 40.8 s"
    assert len(lines) == 6 and [line.split()[0] for line in lines[1:]] == list(STAGES)
    assert "266 pages, 2 empty" in text
    assert "186 to M3 (scanned 8, legacy_font 130, figure 106, sparse table 1; 63 overlapping), 80 normal" in text
    assert "understood 184 (120 from cache), failed 2 (pages 12, 47)" in text
    assert "1036 (figure 44, table 180, text 812)" in text
    assert "1036 texts, 1012 unique: 832 new, 180 cached; batches 167/167; transport retries 2" in text
    assert "1036 upserted, 3 stale deleted; snapshot after write: document 1036 points, collection 3114" in text


def test_format_report_shows_the_failed_stage_and_only_stages_that_started():
    report = known_report(stage="embed", error="ClientError", storage=StoreStats(),
                          seconds={"extract": 6.1, "understand": 28.0, "chunk": 0.4, "embed": 2.0, "total": 36.5})
    text = format_report(report)
    assert text.splitlines()[0].endswith("FAILED at embed (ClientError)")
    assert "  store" not in text and "  embed" in text


def test_format_report_says_when_m3_was_disabled():
    report = known_report(routing=RoutingStats(m3_enabled=False, normal_pages=1, routed_pages=1, reasons={"figure": 1}))
    assert "M3 disabled" in format_report(report) and "understood" not in format_report(report)


@pytest.fixture
def offline_cli(monkeypatch, tmp_path):
    """rag.ingest.main wired to fakes: in-memory Qdrant, hash embedder, fake M3, caches under tmp_path."""
    monkeypatch.setattr(ingest_module, "load_settings", lambda: SimpleNamespace(
        qdrant_url="http://unused", qdrant_api_key=None, gemini_api_key="k", minimax_api_key="k", minimax_base_url="u"))
    monkeypatch.setattr(ingest_module, "QdrantClient", lambda **kw: QdrantClient(":memory:"))
    monkeypatch.setattr(ingest_module, "gemini_embedder", lambda key, task: hash_embed)
    monkeypatch.setattr(ingest_module, "minimax_client", lambda *a, **kw: FakeM3())
    monkeypatch.setattr(ingest_module, "EMBED_CACHE_PATH", tmp_path / "cache" / "e.sqlite")
    monkeypatch.setattr(ingest_module, "UNDERSTAND_CACHE_PATH", tmp_path / "cache" / "u.sqlite")
    return tmp_path


def test_cli_prints_the_summary(offline_cli, sample_pdf, capsys):
    pdf = offline_cli / "sample.pdf"
    pdf.write_bytes(sample_pdf)
    assert ingest_module.main([str(pdf)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("sample.pdf  id=") and "  embed" in out and "understood 1 (0 from cache)" in out


def test_cli_json_is_the_same_report_including_failures(offline_cli, sample_pdf, capsys):
    good, bad = offline_cli / "sample.pdf", offline_cli / "broken.pdf"
    good.write_bytes(sample_pdf)
    bad.write_bytes(b"not a pdf")
    assert ingest_module.main([str(good), str(bad), "--json"]) == 1  # a bad file still fails the run
    captured = capsys.readouterr()
    reports = json.loads(captured.out)  # stdout is pure JSON
    assert [r["source_name"] for r in reports] == ["sample.pdf", "broken.pdf"]
    assert reports[0]["stage"] == "done" and reports[0]["routing"]["understood"] == 1
    assert reports[1]["stage"] == "extract" and reports[1]["error"] == "ValueError"
    assert "FAILED" in captured.err and "sample.pdf  id=" not in captured.out  # no summary mixed into the JSON


# --- audit fixes ------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
NOISES = ("NOISE-python-print", "NOISE-pymupdf-message", "NOISE-native-fd1")
# Run in a child process so stdout is a real file descriptor: PyMuPDF keeps its own stdout handle and
# native code writes to fd 1 directly, and neither is visible to in-process capture such as capsys.
CLI_WITH_STDOUT_NOISE = """
import os, sys
from pathlib import Path
from types import SimpleNamespace

import pymupdf
from qdrant_client import QdrantClient

from rag import ingest
from tests.conftest import build_sample_pdf, hash_embed
from tests.test_understand import FakeM3


def noisy_embedder(texts):
    print("NOISE-python-print")
    pymupdf.message("NOISE-pymupdf-message")
    os.write(1, b"NOISE-native-fd1\\n")
    return hash_embed(texts)


tmp = Path(sys.argv[1])
ingest.load_settings = lambda: SimpleNamespace(
    qdrant_url="http://unused", qdrant_api_key=None, gemini_api_key="k", minimax_api_key="k", minimax_base_url="u")
ingest.QdrantClient = lambda **kw: QdrantClient(":memory:")
ingest.gemini_embedder = lambda key, task: noisy_embedder
ingest.minimax_client = lambda *a, **kw: FakeM3()
ingest.EMBED_CACHE_PATH = tmp / "cache" / "e.sqlite"
ingest.UNDERSTAND_CACHE_PATH = tmp / "cache" / "u.sqlite"
pdf = tmp / "sample.pdf"
pdf.write_bytes(build_sample_pdf())
sys.exit(ingest.main([str(pdf), *sys.argv[2:]]))
"""


def run_cli_subprocess(tmp_path, *flags):
    script = tmp_path / "cli_with_stdout_noise.py"
    script.write_text(CLI_WITH_STDOUT_NOISE, encoding="utf-8")
    env = os.environ | {"PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, str(script), str(tmp_path), *flags], capture_output=True,
                          text=True, encoding="utf-8", env=env, cwd=ROOT, timeout=300)


def test_cli_json_stdout_is_pure_json_when_libraries_and_native_code_write_to_stdout(tmp_path):
    proc = run_cli_subprocess(tmp_path, "--json")
    assert proc.returncode == 0, proc.stderr
    reports = json.loads(proc.stdout)  # the whole stdout parses: nothing else was written to it
    assert [r["stage"] for r in reports] == ["done"] and reports[0]["source_name"] == "sample.pdf"
    for noise in NOISES:
        assert noise not in proc.stdout and noise in proc.stderr  # moved to stderr, not lost


def test_cli_noise_really_reaches_stdout_without_json(tmp_path):
    """Control for the test above: the same noise does land on stdout when --json is not given."""
    proc = run_cli_subprocess(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert all(noise in proc.stdout for noise in NOISES) and "sample.pdf  id=" in proc.stdout


def test_snapshot_counts_are_labelled_and_unknown_until_the_write_completes():
    assert StoreStats() == StoreStats(upserted=0, stale_deleted=None, document_points=None, collection_points=None)
    text = format_report(known_report())
    assert "snapshot after write: document 1036 points, collection 3114" in text


def test_stale_count_failure_never_stops_the_stale_delete(env, monkeypatch):
    ensure_collection(env.client)
    chunks = [Chunk(f"docA-p1-{i}", "docA", "a.pdf", 1, ("S",), "text", f"text {i}") for i in range(3)]
    vectors = [[1.0] * EMBED_DIM] * 3
    replace_document(env.client, "docA", chunks, vectors)

    def count_unavailable(*args, **kwargs):
        raise RuntimeError("count unavailable")

    monkeypatch.setattr(env.client, "count", count_unavailable)
    assert replace_document(env.client, "docA", chunks[:1], vectors[:1]) is None  # count unknown ...
    monkeypatch.undo()
    assert [p["chunk_id"] for p in iter_payloads(env.client)] == ["docA-p1-0"]  # ... but stale points still deleted
    unknown = known_report(storage=StoreStats(upserted=1, stale_deleted=None, document_points=1, collection_points=1))
    assert "1 upserted, ? stale deleted" in format_report(unknown)


def test_partial_store_failure_counts_only_acknowledged_upsert_batches(env, sample_pdf, monkeypatch):
    monkeypatch.setattr(store_module, "UPSERT_BATCH", 2)  # the sample's 4 chunks -> 2 batches
    real_upsert, boom = env.client.upsert, RuntimeError("qdrant write failed")
    calls = []

    def second_batch_fails(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise boom
        return real_upsert(*args, **kwargs)

    monkeypatch.setattr(env.client, "upsert", second_batch_fails)
    report = IngestReport("sample.pdf")
    with pytest.raises(RuntimeError) as raised:
        ingest(env, sample_pdf, report=report)

    assert raised.value is boom
    assert (report.stage, report.error) == ("store", "RuntimeError")
    assert report.storage == StoreStats(upserted=2)  # the first batch landed; nothing after it was measured
    assert "2 upserted before the failure" in format_report(report) and "snapshot" not in format_report(report)


def test_failed_post_write_counts_do_not_fail_a_completed_ingestion(env, sample_pdf, monkeypatch):
    def count_unavailable(*args, **kwargs):
        raise RuntimeError("count unavailable")

    monkeypatch.setattr(env.client, "count", count_unavailable)  # every Qdrant count fails; writes still work
    result = ingest(env, sample_pdf)
    report = result.report
    assert (report.stage, report.error) == ("done", None)
    assert report.storage == StoreStats(upserted=result.chunks)  # stale and both snapshots unknown
    monkeypatch.undo()
    assert len(list(iter_payloads(env.client))) == result.chunks  # the document really was stored
    assert "stale deleted; snapshot after write: document ? points, collection ?" in format_report(report)


def test_successful_store_counts_every_upserted_batch(env, sample_pdf, monkeypatch):
    monkeypatch.setattr(store_module, "UPSERT_BATCH", 3)  # 4 chunks -> batches of 3 and 1
    result = ingest(env, sample_pdf)
    assert result.report.storage.upserted == result.chunks == 4


def test_session_cache_failure_is_reported_as_setup_not_extract(fake_services, sample_pdf, tmp_path):
    collection = session.start_session(fake_services.client)
    fake_services.embed_cache_path = tmp_path  # a directory: SQLite cannot open it
    report = IngestReport("old.pdf", stage="done", error="OldError")  # a reused report is reset first
    with pytest.raises(sqlite3.OperationalError):
        session.ingest_upload(fake_services, collection, sample_pdf, "sample.pdf", report=report)
    assert (report.stage, report.error, report.source_name) == ("setup", "OperationalError", "sample.pdf")


def test_session_touch_failure_after_the_write_is_reported_as_finish_not_done(fake_services, sample_pdf, monkeypatch):
    collection = session.start_session(fake_services.client)
    boom = RuntimeError("could not record activity")

    def touch_fails(*args, **kwargs):
        raise boom

    monkeypatch.setattr(session, "touch_session", touch_fails)
    report = IngestReport("sample.pdf")
    with pytest.raises(RuntimeError) as raised:
        session.ingest_upload(fake_services, collection, sample_pdf, "sample.pdf", report=report)
    assert raised.value is boom
    assert (report.stage, report.error) == ("finish", "RuntimeError")
    assert report.storage.document_points == report.storage.upserted > 0  # the document itself was stored


def test_session_success_ends_done_and_returns_the_callers_report(fake_services, sample_pdf):
    collection = session.start_session(fake_services.client)
    report = IngestReport("sample.pdf")
    result = session.ingest_upload(fake_services, collection, sample_pdf, "sample.pdf", report=report)
    assert result.report is report and (report.stage, report.error) == ("done", None)


def test_reset_restores_defaults_and_keeps_the_object():
    report = known_report(error="RuntimeError", stage="embed")
    same = report
    report.reset("new.pdf")
    assert report is same and report == IngestReport("new.pdf")


def test_reused_report_after_a_failure_does_not_keep_the_error_or_accumulate(env, sample_pdf, monkeypatch):
    monkeypatch.setattr(ingest_module, "embed_texts", partial(embed_texts, batch_size=1))
    report = IngestReport("sample.pdf")
    with pytest.raises(RuntimeError):
        ingest(env, sample_pdf, embedder=FailingSecondBatch(RuntimeError("quota")), report=report)
    assert report.error == "RuntimeError" and report.embedding.batches_attempted == 2

    result = ingest(env, sample_pdf, report=report)  # retry with the same report object
    e = report.embedding
    assert result.report is report and report.error is None and report.stage == "done"
    assert e.texts == result.chunks  # not doubled by the first attempt
    assert (e.cached, e.new, e.batches_attempted) == (1, e.unique - 1, e.unique - 1)  # this run only
