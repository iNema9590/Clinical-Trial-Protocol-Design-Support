import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from agents import (
    extract_eligibility,
    extract_key_assessments,
    extract_objectives,
    extract_soa,
    extract_visit_definitions,
)
from llm import generate
from rag import build_rag_index_from_text
from section_classifier import TARGET_QUERIES, embed

ROUTER_PROMPT = PromptTemplate.from_template(
"""
You are a routing assistant for a clinical trial protocol intelligence system.

Your task is to select the SINGLE most appropriate tool to answer the user’s question.

You must be precise and conservative. 
If the question does not clearly match a specialized extraction tool, choose "rag".

----------------------------------------
TOOL DEFINITIONS
----------------------------------------

1) objectives
Use ONLY when the question is about:
- Study objectives (primary, secondary, exploratory)
- Endpoints linked to objectives
- Relationship between objectives and endpoints

DO NOT use for:
- Procedures
- Visit timing
- Schedule tables

----------------------------------------

2) eligibility
Use ONLY when the question is about:
- Inclusion criteria
- Exclusion criteria
- Participant eligibility rules

----------------------------------------

3) soa
Use ONLY when the question is about:
- Schedule of Activities tables
- Procedures organized by visit
- Visit-by-visit procedure matrices
- Tabular schedule formats

Important:
This is about structured tables mapping procedures to visits.
NOT about describing visits.

----------------------------------------

4) visit_definitions
Use ONLY when the question is about:
- Definitions of visits (Screening, Day 1, Follow-up, etc.)
- Visit timing rules
- Visit windows (± days)
- Triggered visits
- Sequence of visits

Important:
This is about how visits are defined and timed.
NOT about procedures performed at visits.

----------------------------------------

5) key_assessments
Use ONLY when the question is about:
- Assessments (e.g., Safety Assessment, Tumor Assessment)
- Procedures grouped under assessments
- Evaluations and measurements

Important:
This is about assessment → procedure hierarchy.
NOT about objectives.
NOT about visit schedules.

----------------------------------------

6) rag
Use when:
- The question does not clearly match a tool above
- The question spans multiple domains
- The user asks for summary, explanation, or interpretation
- You are uncertain

When unsure → choose "rag".

----------------------------------------

DISAMBIGUATION RULES
----------------------------------------

If the question mentions:

- "primary objective" → objectives
- "endpoint" → objectives
- "inclusion/exclusion" → eligibility
- "schedule of activities" → soa
- "visit window" → visit_definitions
- "Screening visit timing" → visit_definitions
- "procedures under safety assessment" → key_assessments
- "what happens at Day 1?" → soa
- "how is Day 1 defined?" → visit_definitions

----------------------------------------

Question:
{question}

----------------------------------------

Return JSON ONLY in this format:

{{
  "route": "objectives|eligibility|soa|visit_definitions|key_assessments|rag",
  "reason": "one concise sentence explaining why",
  "top_k": 5
}}

Do NOT include explanations outside JSON.
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
        "key_assessments",
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
        parsed_text: str,
        rag_persist_dir: str = "data/rag_index",
        use_existing_rag: bool = True,
    ) -> None:
        self.sections = sections
        self.parsed_text = parsed_text
        self.rag_index = build_rag_index_from_text(
            parsed_text,
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
        elif route == "visit_definitions":
            target = "visit_definitions"
            output = extract_visit_definitions
        else:  # key_assessments
            target = "key_assessments"
            output = extract_key_assessments

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
