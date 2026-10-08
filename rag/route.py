"""Deterministic routing of a chat message before retrieval: no model call, no network.

Retrieval is for questions about document content. Everything else is recognised here first:

- greeting / thanks / ack / farewell / help: answered without the workspace at all.
- followup ("tell me more"): resolved from earlier turns by rag.contextualize before routing; without
  earlier turns it gets a clarifying reply.
- list_documents / page_count: answered from document metadata.
- overview ("what is this about", "what topics are covered", whole-document summaries): answered
  from the document's opening pages (title, contents, introduction), because semantic retrieval
  of "what is this about" finds nothing specific. Still grounded and citation-checked.
- page ("what is on page 72"): answered from that page's chunks, because semantic retrieval cannot
  match a page number.
- vague ("pdf", "what is this?"): a clarifying reply instead of a failed search.
- term (a lone short token such as "ci"): expanded from the documents' own text when possible.
- question: everything else, through the full retrieval pipeline.

Routing only chooses which evidence reaches the generator; it never produces document facts.
Document text never influences it directly: it sees the user's message (or its standalone rewrite,
rag.contextualize) and document names.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

Intent = Literal[
    "greeting", "thanks", "ack", "farewell", "help", "followup",
    "list_documents", "page_count", "overview", "page", "vague", "term", "question",
]
NO_WORKSPACE = frozenset({"greeting", "thanks", "ack", "farewell", "help"})


@dataclass(frozen=True)
class Route:
    intent: Intent
    document: str | None = None  # a workspace document named in the message (its source_name)
    pages: tuple[int, ...] = ()  # for "page"
    term: str | None = None  # for "term"
    query: str | None = None  # rewritten question for retrieval/generation, when the message is a bare topic


GREETING = {"hi", "hello", "hey", "heya", "hiya", "yo", "hola", "namaste", "नमस्ते", "greetings", "morning", "afternoon", "evening", "gm", "sup"}
THANKS = {"thanks", "thank", "thx", "ty", "tysm", "thankyou", "cheers", "appreciated", "appreciate"}
ACK = {"ok", "okay", "k", "kk", "cool", "great", "nice", "awesome", "perfect", "alright", "sure", "gotcha", "understood", "noted", "fine", "good", "yes", "yep", "yeah", "wow"}
FAREWELL = {"bye", "goodbye", "cya", "later", "goodnight"}
FILLER = {"there", "you", "so", "much", "a", "lot", "it", "very", "again", "all", "for", "the", "help", "that", "got", "man", "good", "day", "and", "oh", "ah", "then", "we", "go", "see", "u", "ya", "night"}
SOCIAL = GREETING | THANKS | ACK | FAREWELL

HELP = re.compile(r"^(who are you|what are you|what can you do|how does this work|how do (i|you) use (this|you)|help( me)?|what do you do)$")
HOW_ARE_YOU = re.compile(r"^(how are you( doing)?|how'?s it going|how are things|what'?s up|wh?a?ssup)$")
FOLLOWUP = re.compile(
    r"^(tell me more|more|more details?|explain( that| it| more| this)?|elaborate|can you elaborate|go on|continue|"
    r"why|and|what else|expand( on (that|it))?|explain in detail|keep going|go deeper)$"
)

DOC_WORDS = r"(documents?|docs?|files?|pdfs?|booklets?|reports?|papers?)"
LIST_DOCUMENTS = re.compile(
    rf"\b(what|which|list|show|name)\b.*\b{DOC_WORDS}\b.*\b(uploaded|have|loaded|there|available|added|here|workspace)\b"
    rf"|^(list|show)( me)?( all| the| my)? {DOC_WORDS}$"
    rf"|\bhow many {DOC_WORDS}\b"
    rf"|^(what|which) {DOC_WORDS} (is|are) (this|these|it|loaded|uploaded)$"
    rf"|\bwhat('?s| is| are)? (this|the|these) {DOC_WORDS} (called|named)\b"
    rf"|\b(name|title|file ?name) of (this|the|these) {DOC_WORDS}\b"
)
PAGE_COUNT = re.compile(r"\bhow (many|much) pages?\b|\b(page count|number of pages|how long is (it|this|the))\b")
OVERVIEW = re.compile(
    rf"\bwhat('?s| is| are)? (this|it|these|the {DOC_WORDS}|this {DOC_WORDS})( all)? about\b"
    r"|\bwhat (topics|subjects|chapters|sections|areas|things)\b"
    r"|\b(topics|chapters|sections) (covered|included|in (it|this|the))\b"
    r"|\b(overview|outline|table of contents|contents)\b"
    r"|\b(summari[sz]e|summary|sum up|gist|tl;?dr)\b"
    rf"|\bwhat does (it|this|the {DOC_WORDS}|this {DOC_WORDS}) (cover|contain|include|talk about|discuss)\b"
)
# A page reference, "page 3" or "3rd page"; the number may be invalid (0, negative, huge): respond()
# explains those instead of sending them to retrieval. Numbers are physical PDF pages, starting at 1.
PAGE = re.compile(r"\b(?:page|pg|p)\.?\s*(?:no\.?|number|#)?\s*(-?\d{1,9})\b|\b(\d{1,9})(?:st|nd|rd|th)\s+(?:page|pg)\b")
WHAT_IS_THIS = re.compile(r"\bwhat\b.*\b(this|that|it)\b|^\W*what\W*$")

# Words that carry no topic of their own: a message made only of these is vague.
GENERIC = {
    "pdf", "pdfs", "document", "documents", "doc", "docs", "file", "files", "this", "that", "it", "these", "those", "thing",
    "stuff", "booklet", "report", "paper", "here", "what", "whats", "huh", "hmm", "is", "are", "the", "a", "an", "about",
    "of", "in", "on", "me", "my", "your", "tell", "show", "give", "please", "pls", "can", "you", "do", "does", "one", "uploaded",
}
# Words that frame a request without being its topic ("summarize the key points of X").
FRAMING = GENERIC | {
    "summarize", "summarise", "summary", "overview", "outline", "key", "main", "points", "ideas", "gist", "quick", "short",
    "brief", "give", "an", "all", "whole", "entire", "what", "topics", "covered", "cover", "which", "for", "and", "with",
    "contents", "table", "sum", "up", "tldr", "subjects", "chapters", "sections", "included", "include", "contain", "contains",
    "discuss", "talk", "does", "important", "most", "highlights", "takeaways", "at", "glance", "i", "need", "want", "to",
    "explain", "describe", "list", "covers", "go", "through",
}


def normalize(message: str) -> str:
    """Lowercase, NFKC, curly quotes straightened, punctuation at the ends and repeated spaces dropped."""
    text = unicodedata.normalize("NFKC", message).lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" .,!?;:~-_*\"'()[]")


def words(text: str) -> list[str]:
    # \w alone splits Devanagari at vowel signs and viramas (combining marks), so include that block.
    return re.findall(r"[\w\u0900-\u097f']+", text)


def _only_symbols(message: str) -> bool:
    """Emoji or punctuation only, e.g. "👍" or "?"."""
    return bool(message.strip()) and not any(ch.isalnum() for ch in message)


def _doc_tokens(name: str) -> set[str]:
    stem = re.sub(r"\.pdf$", "", name.lower())
    return {w for w in re.findall(r"[a-z\u0900-\u097f]{4,}", stem)}


def named_document(text: str, names: list[str]) -> str | None:
    """The document the message refers to by (part of) its file name, if exactly one fits best."""
    scores = []
    for name in names:
        stem = re.sub(r"\.pdf$", "", name.lower())
        if stem and stem in text:
            scores.append((100, name))
            continue
        tokens = _doc_tokens(name)
        hits = sum(1 for t in tokens if re.search(rf"\b{re.escape(t)}\b", text))
        if tokens and hits >= min(2, len(tokens)):
            scores.append((hits, name))
    if not scores:
        return None
    scores.sort(reverse=True)
    if len(scores) > 1 and scores[0][0] == scores[1][0]:
        return None
    return scores[0][1]


def classify(message: str, document_names: list[str] | None = None) -> Route:
    """Route one chat message. `document_names` (the workspace's documents) is optional: without it,
    only intents that need no workspace are recognised reliably, which is how callers can skip loading it."""
    names = document_names or []
    if _only_symbols(message):
        return Route("thanks" if "🙏" in message else "ack")
    text = normalize(message)
    tokens = words(text)
    if not tokens:
        return Route("ack")

    if len(tokens) <= 6 and all(t in SOCIAL or t in FILLER for t in tokens) and any(t in SOCIAL for t in tokens):
        for intent, vocabulary in (("thanks", THANKS), ("farewell", FAREWELL), ("greeting", GREETING)):
            if any(t in vocabulary for t in tokens):
                return Route(intent)  # type: ignore[arg-type]
        return Route("ack")
    if HOW_ARE_YOU.match(text):
        return Route("greeting")
    if HELP.match(text):
        return Route("help")
    if FOLLOWUP.match(text):
        return Route("followup")

    document = named_document(text, names)
    doc_words = set(words(re.sub(r"\.pdf$", "", document.lower()))) | {"pdf"} if document else set()

    pages = tuple(dict.fromkeys(int(number or ordinal) for number, ordinal in PAGE.findall(text)))[:3]
    if pages:
        return Route("page", document=document, pages=pages)
    if PAGE_COUNT.search(text):
        return Route("page_count", document=document)
    if LIST_DOCUMENTS.search(text):
        return Route("list_documents", document=document)
    if OVERVIEW.search(text):
        # Only when nothing but framing words and the document's name remain: "summarize the
        # section on compound interest" has a topic of its own and goes to retrieval.
        if all(t in FRAMING or t in doc_words for t in tokens):
            return Route("overview", document=document)

    content = [t for t in tokens if t not in GENERIC and t not in doc_words]
    if not content and len(tokens) <= 6:
        return Route("overview", document=document) if document else Route("vague")
    if len(tokens) == 1:
        (token,) = tokens
        if token.isalpha() and len(token) <= 4:
            return Route("term", term=token)
        return Route("question", query=f"What do the documents say about {message.strip()}?")
    return Route("question", document=document)


STOP = {"the", "and", "for", "are", "was", "with", "can", "has", "have", "its", "not", "but", "you", "all", "any", "our", "who", "how", "this", "that", "from"}


def expand_term(term: str, texts: list[str]) -> str | None:
    """A multi-word phrase in the documents whose initials spell `term` ("ci" -> "Compound Interest").

    Used only to phrase a clarifying question ("Do you mean CI (Compound Interest)?"); the phrase is
    quoted from the documents, never invented. Needs two occurrences, or one written as a definition
    such as "Compound Interest (CI)".
    """
    letters = term.lower()
    if not 2 <= len(letters) <= 4 or not letters.isalpha():
        return None
    phrase_re = re.compile(r"\b" + r"[\s&-]+".join(rf"({c}[a-z]{{2,}})" for c in letters) + r"\b", re.IGNORECASE)
    scores: dict[str, int] = {}
    for text in texts:
        for match in phrase_re.finditer(text):
            parts = match.groups()
            if any(p.lower() in STOP for p in parts):
                continue
            phrase = " ".join(p.capitalize() for p in parts)
            scores[phrase] = scores.get(phrase, 0) + 1
            defined = re.search(rf"{re.escape(match.group(0))}\s*\(\s*{letters}\s*\)|\b{letters}\s*\(\s*{re.escape(match.group(0))}\s*\)", text, re.IGNORECASE)
            if defined:
                scores[phrase] += 5
    best = max(scores.items(), key=lambda kv: kv[1], default=None)
    return best[0] if best and best[1] >= 2 else None
