import pytest

from rag.config import EMBED_DIM
from rag.embed import EmbeddingCache, embed_texts

TASK = "RETRIEVAL_DOCUMENT"


class FakeEmbedder:
    def __init__(self):
        self.batches = []

    def __call__(self, texts):
        self.batches.append(list(texts))
        return [[float(len(t))] + [1.0] * (EMBED_DIM - 1) for t in texts]


def test_embeds_unique_texts_in_batches_and_keeps_order(tmp_path):
    fake, cache = FakeEmbedder(), EmbeddingCache(tmp_path / "e.sqlite")
    texts = ["a", "bb", "a", "ccc", "dddd", "bb"]
    vectors, new = embed_texts(texts, TASK, fake, cache, batch_size=2)
    assert new == 4
    assert [len(b) for b in fake.batches] == [2, 2]
    assert [v[0] for v in vectors] == [1.0, 2.0, 1.0, 3.0, 4.0, 2.0]
    assert all(len(v) == EMBED_DIM for v in vectors)


def test_cache_persists_and_is_never_re_embedded(tmp_path):
    path = tmp_path / "e.sqlite"
    first = EmbeddingCache(path)
    vectors1, _ = embed_texts(["x", "yy"], TASK, FakeEmbedder(), first)
    first.close()

    fake = FakeEmbedder()
    vectors2, new = embed_texts(["yy", "x"], TASK, fake, EmbeddingCache(path))
    assert new == 0 and fake.batches == []
    assert vectors2 == [vectors1[1], vectors1[0]]


def test_only_missing_texts_are_embedded(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    embed_texts(["old"], TASK, FakeEmbedder(), cache)
    fake = FakeEmbedder()
    _, new = embed_texts(["old", "new"], TASK, fake, cache)
    assert new == 1 and fake.batches == [["new"]]


def test_task_type_is_part_of_cache_key(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    embed_texts(["same"], TASK, FakeEmbedder(), cache)
    fake = FakeEmbedder()
    embed_texts(["same"], "RETRIEVAL_QUERY", fake, cache)
    assert fake.batches == [["same"]]


@pytest.mark.parametrize(
    "bad",
    [lambda texts: [], lambda texts: [[0.1] * 3 for _ in texts]],
    ids=["wrong-count", "wrong-dim"],
)
def test_bad_api_response_raises_and_caches_nothing(tmp_path, bad):
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    with pytest.raises(RuntimeError, match="Embedding API returned"):
        embed_texts(["a"], TASK, bad, cache)
    fake = FakeEmbedder()
    embed_texts(["a"], TASK, fake, cache)
    assert fake.batches == [["a"]]
