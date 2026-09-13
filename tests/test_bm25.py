from rag.bm25 import build_bm25, tokenize


def payload(chunk_id, text):
    return {"chunk_id": chunk_id, "text": text}


def test_tokenize_keeps_devanagari_words_whole():
    assert tokenize("कर्मचारियों को भत्ता मिलता है।") == ["कर्मचारियों", "को", "भत्ता", "मिलता", "है"]


def test_tokenize_english_and_hinglish():
    assert tokenize("Paid-Leave: 24 DAYS") == ["paid", "leave", "24", "days"]
    assert tokenize("password kitne din mein badalna hai?") == ["password", "kitne", "din", "mein", "badalna", "hai"]


def test_search_returns_only_matching_chunks_best_first():
    index = build_bm25(
        [
            payload("a", "revenue grew in 2025"),
            payload("b", "office timings and holidays"),
            payload("c", "revenue revenue revenue report for 2025"),
        ]
    )
    results = index.search("revenue 2025", top_k=20)
    assert [p["chunk_id"] for p, _ in results] == ["c", "a"]
    assert results[0][1] >= results[1][1]


def test_search_respects_top_k_and_handles_common_terms():
    # Term present in every chunk: rank_bm25 scores it <= 0, but these are still real matches.
    index = build_bm25([payload(str(i), f"policy number {i}") for i in range(30)])
    assert len(index.search("policy", top_k=20)) == 20


def test_search_hindi_query():
    index = build_bm25([payload("hi", "पासवर्ड हर 90 दिनों में बदलना अनिवार्य है"), payload("en", "leave policy")])
    assert [p["chunk_id"] for p, _ in index.search("पासवर्ड कब बदलना है", top_k=5)] == ["hi"]


def test_search_empty_index_or_query():
    assert build_bm25([]).search("anything", 5) == []
    assert build_bm25([payload("a", "text")]).search("?!", 5) == []
