from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Optional

from .tool import Tool, ToolRegistry

# A "semantic" scorer that needs no third-party embedding library: each tool's
# name + description is turned into a TF-IDF vector and ranked against the query
# by cosine similarity.  Good enough to prune a large tool catalogue down to the
# handful of tools relevant to a user query, cutting prompt tokens and cost.

# Unicode-aware alphanumeric runs.  ``[^\W_]`` = "word char but not underscore",
# so ASCII tokenisation is unchanged (snake_case names still split on '_' into
# sub-words) while Korean/CJK/other scripts are tokenised too — important for a
# codebase whose tool descriptions are written in Korean.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# A scorer maps (query, {tool_name: document_text}) -> {tool_name: score}.
Scorer = Callable[[str, "dict[str, str]"], "dict[str, float]"]


def tfidf_cosine_scores(query: str, documents: dict[str, str]) -> dict[str, float]:
    """Rank *documents* against *query* by TF-IDF cosine similarity.

    Returns a ``{doc_id: score}`` mapping where each score is in ``[0, 1]``.
    Documents with no lexical overlap with the query score ``0.0``.
    """
    if not documents:
        return {}

    doc_tokens = {doc_id: Counter(_tokenize(text)) for doc_id, text in documents.items()}
    n_docs = len(documents)

    # Document frequency: in how many docs each term appears.
    df: Counter[str] = Counter()
    for tokens in doc_tokens.values():
        df.update(tokens.keys())

    # Smoothed IDF so a term in every doc still carries a small positive weight.
    idf = {term: math.log((n_docs + 1) / (freq + 1)) + 1.0 for term, freq in df.items()}

    def vectorize(tokens: Counter[str]) -> dict[str, float]:
        # TF weighting only over terms that exist in the corpus (have IDF).
        return {t: tf * idf[t] for t, tf in tokens.items() if t in idf}

    query_vec = vectorize(Counter(_tokenize(query)))
    query_norm = math.sqrt(sum(w * w for w in query_vec.values()))
    if query_norm == 0.0:
        return {doc_id: 0.0 for doc_id in documents}

    scores: dict[str, float] = {}
    for doc_id, tokens in doc_tokens.items():
        doc_vec = vectorize(tokens)
        doc_norm = math.sqrt(sum(w * w for w in doc_vec.values()))
        if doc_norm == 0.0:
            scores[doc_id] = 0.0
            continue
        # Dot product over the smaller vector's terms.
        dot = sum(weight * doc_vec.get(term, 0.0) for term, weight in query_vec.items())
        scores[doc_id] = dot / (query_norm * doc_norm)
    return scores


class ToolSelector:
    """Selects the most relevant tools for a query (RAG-style tool pruning).

    Instead of sending every registered tool schema to the LLM on each call,
    rank tools by relevance to the user query and forward only the top few.
    This keeps input tokens (and cost) roughly constant as the tool catalogue
    grows.

    Args:
        registry: The ToolRegistry to select from.  Selected tool names must be
            executable by the same registry the agent uses.
        top_k: Maximum number of tools to return.
        min_score: Minimum relevance score (cosine similarity, ``0..1``) for a
            tool to be eligible.  Tools at or below this are excluded.
        fallback_to_all: When no tool clears *min_score*, return the full tool
            list (up to *top_k*) rather than nothing — so a query that happens
            not to lexically overlap any tool still leaves the agent capable.
        scorer: Custom ``(query, {name: text}) -> {name: score}`` function.
            Defaults to :func:`tfidf_cosine_scores`.

    Usage::

        selector = ToolSelector(registry, top_k=3)
        schemas = selector.select_schemas("delete the temp files")
        # → only the most relevant tool schemas
    """

    def __init__(
        self,
        registry: ToolRegistry,
        top_k: int = 3,
        min_score: float = 0.0,
        fallback_to_all: bool = True,
        scorer: Optional[Scorer] = None,
    ) -> None:
        self.registry = registry
        self.top_k = max(0, top_k)
        self.min_score = min_score
        self.fallback_to_all = fallback_to_all
        self._scorer: Scorer = scorer or tfidf_cosine_scores

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(self, query: str) -> list[tuple[Tool, float]]:
        """Return all tools paired with their relevance score, best first."""
        tools = self.registry.all()
        if not tools:
            return []
        documents = {
            name: f"{tool.name} {tool.description}" for name, tool in tools.items()
        }
        scores = self._scorer(query, documents)
        ranked = sorted(
            tools.values(),
            key=lambda t: (scores.get(t.name, 0.0), t.name),
            reverse=True,
        )
        return [(t, scores.get(t.name, 0.0)) for t in ranked]

    def select(self, query: str) -> list[Tool]:
        """Return up to *top_k* tools most relevant to *query*."""
        ranked = self.rank(query)
        if not ranked or self.top_k == 0:
            return []

        eligible = [tool for tool, score in ranked if score > self.min_score]
        if not eligible:
            if self.fallback_to_all:
                eligible = [tool for tool, _ in ranked]
            else:
                return []
        return eligible[: self.top_k]

    def select_schemas(self, query: str) -> list[dict]:
        """Return the JSON schemas of the tools selected for *query*."""
        return [tool.to_schema() for tool in self.select(query)]
