import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from agents import (
    extract_eligibility,
    extract_objectives,
    extract_soa,
    extract_visit_definitions,
)
from llm import generate
from rag import build_rag_index_from_sections
from section_classifier import TARGET_QUERIES, embed

ROUTER_PROMPT = PromptTemplate.from_template(
    """
You are a routing assistant for clinical trial document intelligence.
Choose the best tool for the question based on the tool descriptions below.
Return JSON ONLY.

Tools:
- objectives: Use when the question is about study objectives or endpoints.
- eligibility: Use when the question is about inclusion/exclusion criteria.
- soa: Use when the question is about schedule of activities or procedures by visit.
- visit_definitions: Use when the question is about visit definitions, timing, or visit windows.
- rag: Use for any other question or when unsure.

Question: {question}

Return JSON with this structure:
{{
  "route": "objectives|eligibility|soa|visit_definitions|rag",
  "reason": "short reason",
  "top_k": 5
}}
"""
)


def _safe_json_loads(text: str) -> Dict[str, object]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _serialize_response(obj: object) -> object:
    """Convert Pydantic models to dicts, handle other types."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


def _route_question(question: str) -> Dict[str, object]:
    prompt = ROUTER_PROMPT.format(question=question)
    raw = generate(prompt, temperature=0.0)
    data = _safe_json_loads(raw)
    if not data or "route" not in data:
        return {"route": "rag", "reason": "fallback", "top_k": 5}
    if data.get("route") not in {
        "objectives",
        "eligibility",
        "soa",
        "visit_definitions",
        "rag",
    }:
        data["route"] = "rag"
    if not isinstance(data.get("top_k"), int):
        data["top_k"] = 5
    return data


@dataclass
class SelectionResult:
    sections: List[Tuple[str, float]]
    content: str


class DocumentMultiAgent:
    def __init__(
        self,
        sections: Dict[str, str],
        rag_persist_dir: str = "data/rag_index",
        use_existing_rag: bool = True,
    ) -> None:
        self.sections = sections
        self.rag_index = build_rag_index_from_sections(
            sections,
            persist_dir=rag_persist_dir,
            use_existing=use_existing_rag,
        )
        self.router = RunnableLambda(_route_question)

    def _select_sections(self, target: str, num_sections: int) -> SelectionResult:
        section_titles = list(self.sections.keys())
        section_embeddings = embed(section_titles)

        query_embedding = embed([TARGET_QUERIES[target]])[0]
        scores = torch.matmul(section_embeddings, query_embedding)

        top_scores, top_indices = torch.topk(
            scores, k=min(num_sections, len(section_titles))
        )

        selections = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            selections.append((section_titles[idx], float(score)))

        combined_content = "\n\n".join(
            [self.sections[title] for title, _ in selections]
        )
        return SelectionResult(sections=selections, content=combined_content)

    def _run_specialized_agent(self, route: str, num_sections: int) -> Dict[str, object]:
        if route == "objectives":
            target = "objectives and endpoints"
            output = extract_objectives
        elif route == "eligibility":
            target = "eligibility"
            output = extract_eligibility
        elif route == "soa":
            target = "schedule of activities"
            output = extract_soa
        else:
            target = "visit_definitions"
            output = extract_visit_definitions

        selection = self._select_sections(target, num_sections)
        answer_raw = output(selection.content)
        answer = _serialize_response(answer_raw)
        if isinstance(answer, str):
            answer = _safe_json_loads(answer) or answer

        return {
            "route": route,
            "answer": answer
        }

    def _run_rag(self, question: str, top_k: int) -> Dict[str, object]:
        hits = self.rag_index.query(question, top_k=top_k)
        answer = self.rag_index.answer(question, top_k=top_k)
        return {
            "route": "rag",
            "retrieved": hits,
            "answer": answer,
        }

    def answer(self, question: str, num_sections: int = 2) -> str:
        routing = self.router.invoke(question)
        route = routing.get("route", "rag")
        top_k = routing.get("top_k", 5)

        if route == "rag":
            result = self._run_rag(question, top_k=top_k)
        else:
            result = self._run_specialized_agent(route, num_sections)

        result["question"] = question
        result["routing"] = routing
        return json.dumps(result, ensure_ascii=True)
