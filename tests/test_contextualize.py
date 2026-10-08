"""Follow-up rewriting (rag.contextualize), offline with a recording fake model."""

import pytest

from rag.config import HISTORY_ANSWER_CHARS, HISTORY_TURNS
from rag.contextualize import Turn, build_messages, contextualize
from tests.conftest import RecordingModel

DOCS = ["SRS_Library.pdf", "handbook.pdf"]
HISTORY = [Turn("What is this PDF about?", "An SRS for a library management system [SRS_Library.pdf, Page 1].")]


def test_first_messages_are_never_rewritten():
    model = RecordingModel("should not be used")
    assert contextualize("tell me more", [], DOCS, model) == "tell me more"
    assert contextualize("tell me more", HISTORY, DOCS, None) == "tell me more"  # rewriting disabled
    assert model.calls == []


def test_prompt_carries_documents_recent_turns_and_the_message():
    history = [Turn(f"question {i}", f"answer {i}") for i in range(HISTORY_TURNS + 2)]
    history.append(Turn("What is this PDF about?", "x" * (HISTORY_ANSWER_CHARS + 500)))
    system, user = build_messages("what about page 3?", history, DOCS)
    assert "standalone" in system["content"] and "Never answer" in system["content"]
    text = user["content"]
    assert "SRS_Library.pdf; handbook.pdf" in text and text.endswith("Latest message: what about page 3?")
    assert "question 0" not in text and "question 1" not in text  # only the last HISTORY_TURNS exchanges
    assert "x" * HISTORY_ANSWER_CHARS + " …" in text and "x" * (HISTORY_ANSWER_CHARS + 1) not in text  # long answers cut


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("What is on page 3 of SRS_Library.pdf?", "What is on page 3 of SRS_Library.pdf?"),
        ('<think>resolve "it"</think>\n  "What are the functional requirements in SRS_Library.pdf?"  ',
         "What are the functional requirements in SRS_Library.pdf?"),
    ],
)
def test_rewrite_is_one_clean_line(reply, expected):
    model = RecordingModel(reply)
    assert contextualize("and page 3?", HISTORY, DOCS, model) == expected
    assert len(model.calls) == 1


@pytest.mark.parametrize("reply", [RuntimeError("LLM completion failed: ReadTimeout"), "", "   ", "word " * 200])
def test_any_failure_keeps_the_message_as_typed(reply):
    assert contextualize("tell me more", HISTORY, DOCS, RecordingModel(reply)) == "tell me more"
