"""Tests for RAG-based ToolSelector."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.core.tool import ToolRegistry
from agent.core.tool_selector import ToolSelector, tfidf_cosine_scores


def _registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.register(description="Delete a file from the local filesystem.")
    def delete_file(path: str) -> str:
        return "deleted"

    @reg.register(description="Send an email message to a recipient.")
    def send_email(to: str, body: str) -> str:
        return "sent"

    @reg.register(description="Add two integers and return the sum.")
    def add_numbers(a: int, b: int) -> int:
        return a + b

    @reg.register(description="Fetch the current weather forecast for a city.")
    def get_weather(city: str) -> str:
        return "sunny"

    return reg


class TestTfidfScorer(unittest.TestCase):
    def test_empty_documents(self) -> None:
        self.assertEqual(tfidf_cosine_scores("anything", {}), {})

    def test_relevant_doc_scores_higher(self) -> None:
        docs = {
            "weather": "fetch the current weather forecast for a city",
            "math": "add two integers and return the sum",
        }
        scores = tfidf_cosine_scores("what is the weather forecast", docs)
        self.assertGreater(scores["weather"], scores["math"])

    def test_scores_in_unit_range(self) -> None:
        docs = {"a": "weather forecast city", "b": "add integers sum"}
        for s in tfidf_cosine_scores("weather", docs).values():
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0 + 1e-9)

    def test_query_without_overlap_scores_zero(self) -> None:
        docs = {"a": "weather forecast", "b": "send email"}
        scores = tfidf_cosine_scores("xyzzy plugh", docs)
        self.assertEqual(set(scores.values()), {0.0})


class TestToolSelector(unittest.TestCase):
    def test_selects_most_relevant_tool_first(self) -> None:
        sel = ToolSelector(_registry(), top_k=1)
        selected = sel.select("please remove this file from disk")
        self.assertEqual([t.name for t in selected], ["delete_file"])

    def test_top_k_limits_results(self) -> None:
        sel = ToolSelector(_registry(), top_k=2)
        self.assertLessEqual(len(sel.select("send weather email")), 2)

    def test_select_schemas_returns_schema_dicts(self) -> None:
        sel = ToolSelector(_registry(), top_k=1)
        schemas = sel.select_schemas("delete a file")
        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0]["name"], "delete_file")
        self.assertIn("parameters", schemas[0])

    def test_rank_returns_all_tools_with_scores(self) -> None:
        sel = ToolSelector(_registry())
        ranked = sel.rank("weather")
        self.assertEqual(len(ranked), 4)
        # Descending order by score
        scores = [s for _, s in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(ranked[0][0].name, "get_weather")

    def test_empty_registry_returns_empty(self) -> None:
        sel = ToolSelector(ToolRegistry(), top_k=3)
        self.assertEqual(sel.select("anything"), [])
        self.assertEqual(sel.select_schemas("anything"), [])

    def test_top_k_zero_returns_empty(self) -> None:
        sel = ToolSelector(_registry(), top_k=0)
        self.assertEqual(sel.select("delete file"), [])

    def test_fallback_to_all_when_no_match(self) -> None:
        # Query shares no tokens with any tool description.
        sel = ToolSelector(_registry(), top_k=2, fallback_to_all=True)
        selected = sel.select("zzzz qqqq")
        self.assertEqual(len(selected), 2)  # falls back, capped at top_k

    def test_no_fallback_returns_empty_when_no_match(self) -> None:
        sel = ToolSelector(_registry(), top_k=2, fallback_to_all=False)
        self.assertEqual(sel.select("zzzz qqqq"), [])

    def test_min_score_filters_weak_matches(self) -> None:
        # A very high threshold should exclude everything; with fallback off → empty.
        sel = ToolSelector(
            _registry(), top_k=3, min_score=0.99, fallback_to_all=False
        )
        self.assertEqual(sel.select("weather"), [])

    def test_snake_case_subwords_still_match(self) -> None:
        # 'delete_file' must split on '_' so query token 'delete' matches.
        sel = ToolSelector(_registry(), top_k=1, fallback_to_all=False)
        self.assertEqual([t.name for t in sel.select("delete")], ["delete_file"])

    def test_unicode_korean_tokenization(self) -> None:
        reg = ToolRegistry()

        @reg.register(name="delete_file", description="로컬 파일을 삭제합니다")
        def a(path: str) -> str:
            return "x"

        @reg.register(name="get_weather", description="도시의 날씨 예보를 조회합니다")
        def b(city: str) -> str:
            return "x"

        sel = ToolSelector(reg, top_k=1, fallback_to_all=False)
        # Korean query must rank the Korean-described tool, not fall back to empty.
        self.assertEqual([t.name for t in sel.select("파일을 삭제해줘")], ["delete_file"])
        self.assertEqual([t.name for t in sel.select("날씨 알려줘")], ["get_weather"])

    def test_custom_scorer_is_used(self) -> None:
        def constant_scorer(query, documents):
            # Always rank 'add_numbers' top regardless of query.
            return {name: (1.0 if name == "add_numbers" else 0.1) for name in documents}

        sel = ToolSelector(_registry(), top_k=1, scorer=constant_scorer)
        self.assertEqual(sel.select("delete a file")[0].name, "add_numbers")


if __name__ == "__main__":
    unittest.main()
