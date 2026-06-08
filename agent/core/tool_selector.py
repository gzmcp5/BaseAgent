"""RAG식 도구 선택기 — 질문과 관련된 도구만 골라 LLM에 보낸다(토큰·비용 절감).

도구가 많아질수록 매 호출마다 모든 도구 설명을 LLM에 보내면 입력 토큰이 폭증한다.
ToolSelector는 사용자 질문과 각 도구 설명의 '유사도'를 계산해 상위 몇 개만 추린다.
유사도는 외부 임베딩 라이브러리 없이 TF-IDF + 코사인 유사도로 구한다:
  - TF(단어 빈도): 한 문서에서 그 단어가 얼마나 자주 나오나
  - IDF(역문서 빈도): 그 단어가 드물수록(특이할수록) 높은 가중치
  - 코사인 유사도: 두 벡터(질문 vs 도구설명)가 이루는 각도로 닮은 정도 측정
표준 라이브러리(math/re/collections)만으로 구현해 의존성이 전혀 없다.
"""
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
    # 소문자로 바꾼 뒤 단어 단위로 쪼갠다(대소문자 차이를 무시하기 위함).
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

    # 각 문서를 {단어: 등장횟수} 카운터로 변환(TF의 재료).
    doc_tokens = {doc_id: Counter(_tokenize(text)) for doc_id, text in documents.items()}
    n_docs = len(documents)

    # 문서 빈도(DF): 각 단어가 '몇 개의 문서'에 등장하는지.
    df: Counter[str] = Counter()
    for tokens in doc_tokens.values():
        df.update(tokens.keys())  # 같은 문서 내 중복은 keys()로 1번만 센다.

    # IDF: 흔한 단어일수록 작고, 드문 단어일수록 큰 가중치. +1들은 0 나눗셈·로그(0)을 막는 평활화.
    idf = {term: math.log((n_docs + 1) / (freq + 1)) + 1.0 for term, freq in df.items()}

    def vectorize(tokens: Counter[str]) -> dict[str, float]:
        # TF × IDF로 단어별 가중치 벡터를 만든다. 말뭉치에 없는(IDF가 없는) 단어는 무시.
        return {t: tf * idf[t] for t, tf in tokens.items() if t in idf}

    # 질문을 같은 방식으로 벡터화하고 그 크기(노름)를 구한다.
    query_vec = vectorize(Counter(_tokenize(query)))
    query_norm = math.sqrt(sum(w * w for w in query_vec.values()))
    # 질문에 의미 있는 단어가 하나도 없으면 모든 문서 점수를 0으로(비교 불가).
    if query_norm == 0.0:
        return {doc_id: 0.0 for doc_id in documents}

    scores: dict[str, float] = {}
    for doc_id, tokens in doc_tokens.items():
        doc_vec = vectorize(tokens)
        doc_norm = math.sqrt(sum(w * w for w in doc_vec.values()))
        if doc_norm == 0.0:
            scores[doc_id] = 0.0  # 겹치는 단어가 없는 문서는 점수 0.
            continue
        # 코사인 유사도 = (질문·문서 내적) / (질문 크기 × 문서 크기). 결과는 0~1.
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
        self.registry = registry              # 선택 대상이 되는 도구 레지스트리.
        self.top_k = max(0, top_k)            # 최대 몇 개를 고를지(음수 방지).
        self.min_score = min_score            # 이 점수 이하인 도구는 후보에서 제외.
        self.fallback_to_all = fallback_to_all  # 아무것도 통과 못하면 전체를 줄지 여부.
        self._scorer: Scorer = scorer or tfidf_cosine_scores  # 점수 함수(기본: TF-IDF 코사인).

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(self, query: str) -> list[tuple[Tool, float]]:
        """Return all tools paired with their relevance score, best first."""
        tools = self.registry.all()
        if not tools:
            return []
        # 각 도구의 "이름 + 설명"을 하나의 문서로 삼아 점수를 매긴다.
        documents = {
            name: f"{tool.name} {tool.description}" for name, tool in tools.items()
        }
        scores = self._scorer(query, documents)
        # 점수 내림차순으로 정렬. 점수가 같으면 이름으로 정렬해 결과를 '결정적'으로 만든다.
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

        # min_score를 '초과'하는 도구만 후보로.
        eligible = [tool for tool, score in ranked if score > self.min_score]
        if not eligible:
            # 통과한 도구가 없을 때: fallback이면 전체에서 top_k개, 아니면 빈 목록.
            # (전체를 주는 이유: 질문이 어떤 도구와도 단어가 안 겹쳐도 에이전트가 무력해지지 않게.)
            if self.fallback_to_all:
                eligible = [tool for tool, _ in ranked]
            else:
                return []
        return eligible[: self.top_k]  # 상위 top_k개로 자른다.

    def select_schemas(self, query: str) -> list[dict]:
        """Return the JSON schemas of the tools selected for *query*."""
        # Agent가 실제로 쓰는 진입점: 선택된 도구들의 LLM용 스키마만 돌려준다.
        return [tool.to_schema() for tool in self.select(query)]
