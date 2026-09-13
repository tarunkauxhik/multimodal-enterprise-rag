"""Live multilingual retrieval test against real Gemini, Qdrant and Jina.

Opt-in (costs API calls): RAG_LIVE_TESTS=1 uv run pytest tests/test_live_retrieval.py -v -s
Uses a temporary Qdrant collection and a temporary embedding cache; both are removed afterwards.
Hindi chunks are loaded directly so retrieval quality is tested independently of PDF extraction.
"""

import os
import uuid

import pytest
from qdrant_client import QdrantClient

from rag.chunk import Chunk
from rag.config import EMBED_TASK_DOCUMENT, EMBED_TASK_QUERY, load_settings
from rag.embed import EmbeddingCache, embed_texts, gemini_embedder
from rag.rerank import jina_reranker
from rag.retrieve import Retriever
from rag.store import ensure_collection, replace_document

pytestmark = pytest.mark.skipif(
    os.environ.get("RAG_LIVE_TESTS") != "1", reason="set RAG_LIVE_TESTS=1 to call Gemini, Jina and Qdrant"
)

EN_BOOK, HI_BOOK, REPORT = "Employee Handbook", "कर्मचारी पुस्तिका", "Annual Report"

CORPUS = [
    (1, (EN_BOOK, "Leave Policy"), "Full-time employees receive 24 days of paid annual leave per calendar year. Up to 10 unused leave days can be carried forward to the next year."),
    (1, (EN_BOOK, "Sick Leave"), "Employees are entitled to 12 days of paid sick leave. A medical certificate is required for absences longer than three consecutive days."),
    (2, (HI_BOOK, "यात्रा भत्ता नीति"), "व्यावसायिक यात्रा के दौरान कर्मचारियों को प्रतिदिन 3000 रुपये का दैनिक भत्ता मिलता है। होटल का खर्च कंपनी सीधे चुकाती है।"),
    (2, (HI_BOOK, "कार्यालय समय"), "कार्यालय का समय सुबह 9:30 बजे से शाम 6:00 बजे तक है। शनिवार और रविवार को अवकाश रहता है।"),
    (3, (HI_BOOK, "सूचना सुरक्षा"), "सभी कर्मचारियों को हर 90 दिनों में अपना पासवर्ड बदलना अनिवार्य है। पासवर्ड किसी के साथ साझा नहीं करना चाहिए।"),
    (3, (EN_BOOK, "IT Support"), "For laptop or VPN problems, raise a ticket on the IT helpdesk portal. Critical issues are resolved within 4 hours."),
    (4, (REPORT, "Quarterly Revenue"), "|Quarter|Revenue (INR crore)|\n|---|---|\n|Q1|120|\n|Q2|135|\n|Q3|150|\n|Q4|170|"),
    (4, (REPORT, "Outlook"), "The company plans to open two new offices in Pune and Hyderabad next year and hire 300 engineers."),
    (5, ("वार्षिक रिपोर्ट", "ग्राहक संतुष्टि"), "इस वर्ष ग्राहक संतुष्टि स्कोर 82 से बढ़कर 91 हो गया।"),
    (5, (EN_BOOK, "Health Insurance"), "All employees and their dependents are covered by group health insurance up to 5 lakh rupees per year."),
    (6, (HI_BOOK, "वर्क फ्रॉम होम"), "कर्मचारी सप्ताह में अधिकतम दो दिन घर से काम कर सकते हैं, इसके लिए प्रबंधक की अनुमति आवश्यक है।"),
    (6, (EN_BOOK, "Expense Reimbursement"), "Submit expense claims with original receipts within 30 days. Reimbursements are paid with the next monthly salary."),
]

QUERIES = [
    ("english", "How many days of paid annual leave do employees get?", "Leave Policy"),
    ("english->hindi", "What is the daily allowance for business travel?", "यात्रा भत्ता नीति"),
    ("hindi", "यात्रा के दौरान दैनिक भत्ता कितना मिलता है?", "यात्रा भत्ता नीति"),
    ("hindi->english", "कंपनी की तिमाही आय कितनी रही?", "Quarterly Revenue"),
    ("hinglish->hindi", "password kitne din mein badalna padta hai", "सूचना सुरक्षा"),
    ("hinglish->hindi", "ghar se kaam hafte mein kitne din kar sakte hain", "वर्क फ्रॉम होम"),
    ("hinglish->english", "sick leave ke liye medical certificate kab chahiye", "Sick Leave"),
]


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    settings = load_settings()
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    collection = f"live_test_{uuid.uuid4().hex[:8]}"
    cache = EmbeddingCache(tmp_path_factory.mktemp("cache") / "embeddings.sqlite")
    chunks = [
        Chunk(f"livedoc-p{page}-{i}", "livedoc", "policies.pdf", page, path, "text", text)
        for i, (page, path, text) in enumerate(CORPUS)
    ]
    try:
        ensure_collection(client, collection)
        vectors, _ = embed_texts(
            [c.text for c in chunks], EMBED_TASK_DOCUMENT, gemini_embedder(settings.gemini_api_key, EMBED_TASK_DOCUMENT), cache
        )
        replace_document(client, "livedoc", chunks, vectors, collection)
        yield Retriever(
            client,
            gemini_embedder(settings.gemini_api_key, EMBED_TASK_QUERY),
            cache,
            jina_reranker(settings.jina_api_key),
            collection=collection,
        )
    finally:
        client.delete_collection(collection)
        cache.close()
        client.close()


@pytest.mark.parametrize("kind, query, expected_section", QUERIES, ids=[f"{k}-{i}" for i, (k, _, _) in enumerate(QUERIES)])
def test_multilingual_query_retrieves_expected_section(retriever, kind, query, expected_section):
    hits = retriever.retrieve(query)
    print(f"\n[{kind}] {query}")
    for rank, h in enumerate(hits, start=1):
        print(
            f"  {rank}. p{h.payload['page_number']} {h.payload['section_path'][-1]} "
            f"rerank={h.rerank_score:.3f} dense={h.dense_rank} bm25={h.bm25_rank}"
        )

    assert len(hits) == 5
    assert hits[0].payload["section_path"][-1] == expected_section
    for h in hits:
        assert h.payload["document_id"] == "livedoc" and h.payload["source_name"] == "policies.pdf"
        assert isinstance(h.payload["page_number"], int) and h.rrf_score is not None
        if h.dense_score is not None:
            assert -1.0 <= h.dense_score <= 1.0
