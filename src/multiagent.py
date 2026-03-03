import json
import ast
import os
import re
import torch
from typing import Annotated, Literal, TypedDict, Any
from dataclasses import dataclass
import pandas as pd

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from operator import add

from agents import (
    extract_eligibility,
    extract_key_assessments,
    extract_objectives,
    extract_soa,
    extract_visit_definitions,
)
from llm import generate
from rag import ClinicalProtocolRAG
from section_classifier import TARGET_QUERIES, embed


# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """State passed through the supervisor agent graph."""
    messages: Annotated[list[BaseMessage], add]
    query: str
    route: Literal[
        "objectives and endpoints",
        "eligibility",
        "eligibility check",
        "schedule of activities",
        "visit definitions",
        "key assessments",
        "rag",
    ]
    routing_info: dict


# ============================================================================
# SUPERVISOR PROMPT
# ============================================================================

SUPERVISOR_PROMPT = PromptTemplate.from_template(
    """You are a routing assistant for a clinical trial protocol intelligence system.

Your task is to select the SINGLE most appropriate tool to answer the user's question.

You must be precise and conservative. 
If the question does not clearly match a specialized extraction tool, choose "rag".

----------------------------------------
TOOL DEFINITIONS
----------------------------------------

1) objectives and endpoints
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

2b) eligibility check
Use ONLY when the question is about:
- Validating patient dataset against eligibility criteria
- Eligibility compliance check
- Finding non-eligible / non-compliant patients
- Data validation using protocol inclusion/exclusion rules

----------------------------------------

3) schedule of activities
Use ONLY when the question is about:
- Schedule of Activities tables
- Procedures organized by visit
- Visit-by-visit procedure matrices
- Tabular schedule formats

Important:
This is about structured tables mapping procedures to visits.
NOT about describing visits.

----------------------------------------

4) visit definitions
Use ONLY when the question is about:
- Listing the visit definitions
- structured output of visit definitions

----------------------------------------

5) key assessments
Use ONLY when the question is about:
- Assessments (e.g., Safety Assessment, Tumor Assessment)
- Procedures grouped under assessments
- Evaluations and measurements

----------------------------------------

6) rag
Use when:
- The question does not clearly match a tool above
- The question spans multiple domains
- The user asks for summary, explanation, or interpretation
- You are uncertain

----------------------------------------

DISAMBIGUATION RULES
----------------------------------------

ROUTING GUIDE BY KEYWORDS:

**OBJECTIVES AND ENDPOINTS (if query contains):**
- "objective" OR "objectives" (primary/secondary/exploratory)
- "endpoint" OR "endpoints"
- "goal" + "study"
- "aim" + "study"

**ELIGIBILITY (if query contains):**
- "inclusion" OR "exclusion" 
- "eligible" OR "eligibility"
- "participant selection"
- "criteria" (but NOT "eligibility check")

**ELIGIBILITY CHECK (if query contains):**
- "validation" + "eligibility"
- "validate" + "patient" OR "patients"
- "check" + "patient" OR "patients"
- "compliance" + "eligibility"
- "non-eligible" OR "ineligible"
- "meet eligibility requirement"

**SCHEDULE OF ACTIVITIES / SOA (if query contains):**
- "schedule of activities"
- "procedures" + ("at" OR "during") + visit/day
- "what happens" (at visit/day/week)
- "what is done" (at visit/day/week)
- "procedures" + ("visit" OR "day" OR "week")
- "procedure matrix"

**VISIT DEFINITIONS (if query contains):**
- "visit definition" OR "visit definitions"
- "how is" + visit mentioned
- "when is" + visit mentioned
- "visit" + ("window" OR "timing" OR "duration")
- "screening" + ("window" OR "period" OR "defined")
- "baseline" + ("timing" OR "window")
- "final visit"

**KEY ASSESSMENTS (if query contains):**
- "assessment" OR "assessments"
- "procedure" + "safety" OR "efficacy"
- "what is assessed"
- "what is measured"
- "evaluation"
- "efficacy assessment"
- "safety assessment"

**RAG/FALLBACK (if unsure or):**
- Question spans multiple domains
- Request for summary/overview
- General protocol questions
- Does not match above patterns

When query matches multiple categories, pick the most specific one.

----------------------------------------

Question:
{question}

----------------------------------------

Return JSON ONLY in this format:

{{
    "route": "objectives and endpoints|eligibility|eligibility check|schedule of activities|visit definitions|key assessments|rag",
  "reason": "one concise sentence explaining why"
}}

Do NOT include explanations outside JSON.
"""
)


def _parse_router_output(raw: str) -> dict[str, Any]:
    """Parse router JSON robustly (handles fenced JSON and extra text)."""
    if not raw:
        return {}

    text = raw.strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    first_json = re.search(r"\{.*\}", text, re.DOTALL)
    if first_json:
        candidate = first_json.group(0).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, SyntaxError):
                pass

    return {}


# ============================================================================
# AGENT CONFIGURATION
# ============================================================================

FUNCTIONS = {
    "objectives and endpoints":extract_objectives,
    "eligibility": extract_eligibility,
    "schedule of activities":extract_soa,
    "visit definitions": extract_visit_definitions,
    "key assessments": extract_key_assessments
}

# ============================================================================
# AGENTS WRAPPED AS TOOLS
# ============================================================================

@dataclass
class SelectionResult:
    sections: list[tuple[str, float]]
    content: str


class DocumentMultiAgentCore:
    """Core functionality for document analysis with specialized agents and RAG."""
    
    def __init__(
        self,
        sections: dict[str, str],
        parsed_text: str,
        rag_persist_dir: str = "../data/rag_index",
        patient_data_path: str = "../data/synthetic_patient_data.csv",
    ) -> None:
        self.sections = sections
        self.parsed_text = parsed_text
        self.rag_index = ClinicalProtocolRAG(parsed_text)
        self.patient_data_path = patient_data_path

    def _resolve_patient_data_path(self) -> str:
        candidates = [
            self.patient_data_path,
            "../data/synthetic_patient_data.csv",
        ]
        for path in candidates:
            if path and os.path.exists(path):
                return path
        raise FileNotFoundError(
            "Patient dataset not found. Expected one of: "
            f"{', '.join(candidates)}"
        )

    def _extract_numbers(self, text: str) -> list[float]:
        return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]

    def _evaluate_rule(
        self,
        rule: dict[str, Any],
        df: pd.DataFrame,
    ) -> tuple[pd.Series | None, str]:
        """
        Evaluate a structured eligibility rule against the dataframe.
        Dynamically determines field type from actual data.
        
        Args:
            rule: Dict with keys: text, field, operator, value, evaluable
            df: Patient dataframe
        
        Returns:
            Tuple of (boolean mask, parsed rule description) or (None, error message)
        """
        if not rule.get("evaluable", False):
            return None, "Rule not machine-evaluable"
        
        field = rule.get("field")
        operator = rule.get("operator")
        value = rule.get("value")
        
        if not field or not operator or value is None:
            return None, "Missing required rule components"
        
        if field not in df.columns:
            return None, f"Field {field} not found in dataset"
        
        try:
            col = df[field]
            
            # Detect field type from actual data
            is_boolean_field = False
            is_categorical_field = False
            is_numeric_field = False
            
            # Check if column contains boolean-like values ("True"/"False" strings or actual bools)
            unique_vals = col.astype(str).str.strip().unique()
            if set(unique_vals).issubset({"True", "False"}):
                is_boolean_field = True
            elif col.dtype in [bool, object] and len(unique_vals) <= 10:
                # Categorical if <= 10 unique values
                is_categorical_field = True
            else:
                is_numeric_field = True
            
            # Handle different operators
            if operator == "between":
                # Value format: "min,max"
                if not is_numeric_field:
                    return None, f"Range operator 'between' requires numeric field"
                parts = str(value).split(",")
                if len(parts) != 2:
                    return None, f"Invalid range format: {value}"
                low, high = float(parts[0]), float(parts[1])
                mask = col.between(low, high)
                return mask, f"{field} between {low} and {high}"
            
            elif operator == ">=":
                if not is_numeric_field:
                    return None, f"Operator '>=' requires numeric field"
                threshold = float(value)
                mask = col >= threshold
                return mask, f"{field} >= {threshold}"
            
            elif operator == "<=":
                # Check if value contains ULN reference
                if "ULN" in str(value).upper():
                    if "ULN" not in df.columns:
                        return None, "ULN column not found in dataset"
                    # Parse multiplier (e.g., "2*ULN" → 2)
                    multiplier_match = re.search(r"([\d.]+)\s*\*?\s*ULN", str(value), re.IGNORECASE)
                    multiplier = float(multiplier_match.group(1)) if multiplier_match else 1.0
                    mask = col <= multiplier * df["ULN"]
                    return mask, f"{field} <= {multiplier} × ULN"
                threshold = float(value)
                mask = col <= threshold
                return mask, f"{field} <= {threshold}"
            
            elif operator == ">":
                if not is_numeric_field:
                    return None, f"Operator '>' requires numeric field"
                threshold = float(value)
                mask = col > threshold
                return mask, f"{field} > {threshold}"
            
            elif operator == "<":
                # Check if value contains ULN reference
                if "ULN" in str(value).upper():
                    if "ULN" not in df.columns:
                        return None, "ULN column not found in dataset"
                    multiplier_match = re.search(r"([\d.]+)\s*\*?\s*ULN", str(value), re.IGNORECASE)
                    multiplier = float(multiplier_match.group(1)) if multiplier_match else 1.0
                    mask = col < multiplier * df["ULN"]
                    return mask, f"{field} < {multiplier} × ULN"
                threshold = float(value)
                mask = col < threshold
                return mask, f"{field} < {threshold}"
            
            elif operator == "==":
                if is_boolean_field:
                    # Boolean comparison
                    expected_bool = str(value).strip() in ["True", "true", "1", "yes", "Yes"]
                    csv_bool_col = col.astype(str).str.strip().isin(["True", "true", "1", "yes", "Yes"])
                    mask = csv_bool_col == expected_bool
                    return mask, f"{field} == {value}"
                else:
                    # String/categorical comparison (case-sensitive)
                    mask = col.astype(str).str.strip() == str(value).strip()
                    return mask, f"{field} == {value}"
            
            elif operator == "!=":
                if is_boolean_field:
                    # Boolean comparison
                    expected_bool = str(value).strip() in ["True", "true", "1", "yes", "Yes"]
                    csv_bool_col = col.astype(str).str.strip().isin(["True", "true", "1", "yes", "Yes"])
                    mask = csv_bool_col != expected_bool
                    return mask, f"{field} != {value}"
                else:
                    # String/categorical comparison (case-sensitive)
                    mask = col.astype(str).str.strip() != str(value).strip()
                    return mask, f"{field} != {value}"
            
            elif operator == "in":
                # Value format: "val1,val2,val3"
                valid_values = [v.strip() for v in str(value).split(",")]
                mask = col.astype(str).str.strip().isin(valid_values)
                return mask, f"{field} in [{', '.join(valid_values)}]"
            
            else:
                return None, f"Unsupported operator: {operator}"
        
        except Exception as e:
            return None, f"Evaluation error: {str(e)}"

    def eligibility_check_agent(self, query: str) -> dict[str, Any]:
        """
        1) Extract structured eligibility criteria with rules
        2) Apply rule checks on patient dataset
        3) Return non-compliant patients
        """
        eligibility_result = self.extract_eligibility_agent(query)
        eligibility = eligibility_result.get("answer", {})
        inclusion_rules = eligibility.get("inclusion", []) if isinstance(eligibility, dict) else []
        exclusion_rules = eligibility.get("exclusion", []) if isinstance(eligibility, dict) else []

        dataset_path = self._resolve_patient_data_path()
        df = pd.read_csv(dataset_path)
        if "PATIENT_ID" not in df.columns:
            df = df.reset_index().rename(columns={"index": "PATIENT_ID"})

        reasons_by_patient: dict[Any, list[str]] = {
            patient_id: [] for patient_id in df["PATIENT_ID"].tolist()
        }
        evaluated_rules: list[dict[str, Any]] = []
        unevaluated_criteria: list[dict[str, str]] = []

        # Process inclusion criteria
        for rule in inclusion_rules:
            if not isinstance(rule, dict):
                continue
            
            mask, parsed_rule = self._evaluate_rule(rule, df)
            if mask is None:
                unevaluated_criteria.append({
                    "type": "inclusion",
                    "criterion": rule.get("text", ""),
                    "reason": parsed_rule
                })
                continue
            
            evaluated_rules.append({
                "type": "inclusion",
                "criterion": rule.get("text", ""),
                "parsed_rule": parsed_rule,
                "field": rule.get("field"),
                "operator": rule.get("operator"),
                "value": rule.get("value"),
            })
            
            # Patients who don't meet inclusion criteria
            failed_ids = df.loc[~mask, "PATIENT_ID"].tolist()
            for patient_id in failed_ids:
                reasons_by_patient[patient_id].append(
                    f"Failed inclusion: {rule.get('text', '')}"
                )

        # Process exclusion criteria
        for rule in exclusion_rules:
            if not isinstance(rule, dict):
                continue
            
            mask, parsed_rule = self._evaluate_rule(rule, df)
            if mask is None:
                unevaluated_criteria.append({
                    "type": "exclusion",
                    "criterion": rule.get("text", ""),
                    "reason": parsed_rule
                })
                continue
            
            evaluated_rules.append({
                "type": "exclusion",
                "criterion": rule.get("text", ""),
                "parsed_rule": parsed_rule,
                "field": rule.get("field"),
                "operator": rule.get("operator"),
                "value": rule.get("value"),
            })
            
            # Patients who meet exclusion criteria (should be excluded)
            triggered_ids = df.loc[mask, "PATIENT_ID"].tolist()
            for patient_id in triggered_ids:
                reasons_by_patient[patient_id].append(
                    f"Triggered exclusion: {rule.get('text', '')}"
                )

        non_eligible = []
        for patient_id, reasons in reasons_by_patient.items():
            if reasons:
                row = df.loc[df["PATIENT_ID"] == patient_id].iloc[0].to_dict()
                non_eligible.append({
                    "patient_id": patient_id,
                    "reasons": reasons,
                    "patient_record": row,
                })

        return {
            "source": "eligibility check",
            "dataset_path": dataset_path,
            "total_patients": int(len(df)),
            "non_eligible_count": int(len(non_eligible)),
            "non_eligible_patients": non_eligible,
            "evaluated_rules": evaluated_rules,
            "unevaluated_criteria": unevaluated_criteria,
            "structured_eligibility": eligibility,
            "selected_sections": eligibility_result.get("selected_sections", []),
        }
    
    def _select_sections(self, target: str, num_sections: int = 3) -> SelectionResult:
        """Select most relevant sections based on target query."""
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
    
    def _serialize_response(self, obj: Any) -> Any:
        """Convert Pydantic models to dicts."""
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        return obj
    
    def run_extraction(
        self,
        agent_name: str,
        num_sections: int = 3,
    ) -> dict[str, Any]:
        """
        Unified agent extraction method.
        
        Args:
            agent_name: One of "objectives", "eligibility", "schedule of activities", "visit definitions", "key assessments"
            num_sections: Number of top sections to retrieve
        
        Returns:
            Dict with source and answer keys
        """
        if agent_name not in FUNCTIONS:
            raise ValueError(
                f"Unknown agent: {agent_name}. "
                f"Must be one of {list(FUNCTIONS.keys())}"
            )
        
        extraction_func = FUNCTIONS[agent_name]
        
        # Get top relevant sections
        selection = self._select_sections(agent_name, num_sections=num_sections)
        
        # Run extraction
        answer_raw = extraction_func(selection.content)
        answer = self._serialize_response(answer_raw)
        
        return {
            "source": agent_name,
            "answer": answer,
            "selected_sections": selection.sections,
        }
    
    def extract_objectives_agent(self, query: str) -> dict[str, Any]:
        """Extract study objectives and endpoints."""
        return self.run_extraction("objectives and endpoints", num_sections=3)
    
    def extract_eligibility_agent(self, query: str) -> dict[str, Any]:
        """Extract inclusion and exclusion criteria."""
        return self.run_extraction("eligibility", num_sections=3)
    
    def extract_soa_agent(self, query: str) -> dict[str, Any]:
        """Extract Schedule of Activities."""
        return self.run_extraction("schedule of activities", num_sections=3)
    
    def extract_visit_definitions_agent(self, query: str) -> dict[str, Any]:
        """Extract visit definitions and timing."""
        return self.run_extraction("visit definitions", num_sections=3)
    
    def extract_key_assessments_agent(self, query: str) -> dict[str, Any]:
        """Extract key assessments and procedures."""
        return self.run_extraction("key assessments", num_sections=3)
    
    def rag_agent(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Use RAG to answer the query."""
        docs = self.rag_index.retrieve(query, k=top_k)
        answer = self.rag_index.answer(query, k=top_k)
        return {
            "source": "rag",
            "answer": answer,
            "retrieved_context": [doc.page_content for doc in docs],
        }


# ============================================================================
# LANGGRAPH SUPERVISOR AGENT
# ============================================================================

class SupervisorMultiAgent:
    """LangGraph-based supervisor agent for routing queries."""
    
    def __init__(
        self,
        sections: dict[str, str],
        parsed_text: str,
        rag_persist_dir: str = "../data/rag_index",
        patient_data_path: str = "../data/synthetic_patient_data.csv",
    ):
        self.core = DocumentMultiAgentCore(
            sections=sections,
            parsed_text=parsed_text,
            rag_persist_dir=rag_persist_dir,
            patient_data_path=patient_data_path,
        )
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph state machine."""
        workflow = StateGraph(AgentState)
        
        # Add supervisor node
        workflow.add_node("supervisor", self._supervisor_node)
        
        # Add agent nodes
        workflow.add_node("objectives and endpoints", self._objectives_node)
        workflow.add_node("eligibility", self._eligibility_node)
        workflow.add_node("eligibility check", self._eligibility_check_node)
        workflow.add_node("schedule of activities", self._soa_node)
        workflow.add_node("visit definitions", self._visit_definitions_node)
        workflow.add_node("key assessments", self._key_assessments_node)
        workflow.add_node("rag", self._rag_node)
        
        # Add final node
        workflow.add_node("final", self._final_node)
        
        # Entry point
        workflow.set_entry_point("supervisor")
        
        # Routing from supervisor
        workflow.add_conditional_edges(
            "supervisor",
            self._route_supervisor,
            {
                "objectives and endpoints": "objectives and endpoints",
                "eligibility": "eligibility",
                "eligibility check": "eligibility check",
                "schedule of activities": "schedule of activities",
                "visit definitions": "visit definitions",
                "key assessments": "key assessments",
                "rag": "rag",
            }
        )
        
        # All agent nodes go to final
        workflow.add_edge("objectives and endpoints", "final")
        workflow.add_edge("eligibility", "final")
        workflow.add_edge("eligibility check", "final")
        workflow.add_edge("schedule of activities", "final")
        workflow.add_edge("visit definitions", "final")
        workflow.add_edge("key assessments", "final")
        workflow.add_edge("rag", "final")
        
        # Final node ends
        workflow.add_edge("final", END)
        
        return workflow.compile()
    
    def _supervisor_node(self, state: AgentState) -> AgentState:
        """Supervisor node: route query to appropriate agent."""
        query = state["query"]

        normalized_query = query.lower()
        if (
            "validation" in normalized_query
            and ("eligibility" in normalized_query or "eligigbility" in normalized_query)
            and "check" in normalized_query
        ):
            routing_data = {
                "route": "eligibility check",
                "reason": "Detected explicit eligibility validation/compliance request.",
            }
            route = "eligibility check"
            routing_msg = AIMessage(content=f"Routing to: {route}")
            return {
                **state,
                "messages": state.get("messages", []) + [routing_msg],
                "route": route,
                "routing_info": routing_data,
            }
        
        # Get routing decision
        prompt = SUPERVISOR_PROMPT.format(question=query)
        raw = generate(prompt, temperature=0.0)

        routing_data = _parse_router_output(raw)
        if not routing_data:
            routing_data = {
                "route": "rag",
                "reason": "routing parse failed",
                "raw_router_output": raw,
            }

        route = str(routing_data.get("route", "rag")).strip().lower()
        if route not in [
            "objectives and endpoints",
            "eligibility",
            "eligibility check",
            "schedule of activities",
            "visit definitions",
            "key assessments",
            "rag",
        ]:
            route = "rag"
        routing_data["route"] = route
        
        # Add routing to messages
        routing_msg = AIMessage(content=f"Routing to: {route}")
        
        return {
            **state,
            "messages": state.get("messages", []) + [routing_msg],
            "route": route,
            "routing_info": routing_data,
        }
    
    def _route_supervisor(self, state: AgentState) -> str:
        """Determine which agent node to visit."""
        return state.get("route", "rag")
    
    def _objectives_node(self, state: AgentState) -> AgentState:
        """Execute objectives extraction."""
        result = self.core.extract_objectives_agent(state["query"])
        msg = AIMessage(content=json.dumps(result))
        return {
            **state,
            "messages": state.get("messages", []) + [msg],
        }
    
    def _eligibility_node(self, state: AgentState) -> AgentState:
        """Execute eligibility extraction."""
        result = self.core.extract_eligibility_agent(state["query"])
        msg = AIMessage(content=json.dumps(result))
        return {
            **state,
            "messages": state.get("messages", []) + [msg],
        }

    def _eligibility_check_node(self, state: AgentState) -> AgentState:
        """Execute eligibility extraction + dataset compliance check."""
        result = self.core.eligibility_check_agent(state["query"])
        msg = AIMessage(content=json.dumps(result))
        return {
            **state,
            "messages": state.get("messages", []) + [msg],
        }
    
    def _soa_node(self, state: AgentState) -> AgentState:
        """Execute Schedule of Activities extraction."""
        result = self.core.extract_soa_agent(state["query"])
        msg = AIMessage(content=json.dumps(result))
        return {
            **state,
            "messages": state.get("messages", []) + [msg],
        }
    
    def _visit_definitions_node(self, state: AgentState) -> AgentState:
        """Execute visit definitions extraction."""
        result = self.core.extract_visit_definitions_agent(state["query"])
        msg = AIMessage(content=json.dumps(result))
        return {
            **state,
            "messages": state.get("messages", []) + [msg],
        }
    
    def _key_assessments_node(self, state: AgentState) -> AgentState:
        """Execute key assessments extraction."""
        result = self.core.extract_key_assessments_agent(state["query"])
        msg = AIMessage(content=json.dumps(result))
        return {
            **state,
            "messages": state.get("messages", []) + [msg],
        }
    
    def _rag_node(self, state: AgentState) -> AgentState:
        """Execute RAG query."""
        result = self.core.rag_agent(state["query"])
        msg = AIMessage(content=json.dumps(result))
        return {
            **state,
            "messages": state.get("messages", []) + [msg],
        }
    
    def _final_node(self, state: AgentState) -> AgentState:
        """Final node: format output."""
        return state
    
    def answer(self, question: str) -> str:
        """Process a question through the supervisor agent."""
        initial_state: AgentState = {
            "messages": [HumanMessage(content=question)],
            "query": question,
            "route": "rag",
            "routing_info": {},
        }
        
        final_state = self.graph.invoke(initial_state)
        
        # Extract the agent's response (last message)
        if final_state["messages"]:
            last_msg = final_state["messages"][-1]
            try:
                result = json.loads(last_msg.content)
            except (json.JSONDecodeError, AttributeError):
                result = {"answer": str(last_msg.content)}
        else:
            result = {"answer": "No response"}
        
        # Add metadata
        result["question"] = question
        result["route"] = final_state.get("route", "unknown")
        result["routing_info"] = final_state.get("routing_info", {})
        
        return json.dumps(result, ensure_ascii=True)
    
    async def astreaming_answer(self, question: str):
        """Stream the supervisor agent's respsonse."""
        initial_state: AgentState = {
            "messages": [HumanMessage(content=question)],
            "query": question,
            "route": "rag",
            "routing_info": {},
        }
        
        # Stream events from the graph
        async for event in self.graph.astream_events(initial_state, version="v2"):
            yield event


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

class DocumentMultiAgent:
    """Backward compatible wrapper for the new supervisor agent."""
    
    def __init__(
        self,
        sections: dict[str, str],
        parsed_text: str,
        rag_persist_dir: str = "../data/rag_index",
        patient_data_path: str = "../data/synthetic_patient_data.csv",
    ):
        self.supervisor = SupervisorMultiAgent(
            sections=sections,
            parsed_text=parsed_text,
            rag_persist_dir=rag_persist_dir,
            patient_data_path=patient_data_path,
        )
    
    def answer(self, question: str) -> str:
        """Process a question through the supervisor agent."""
        return self.supervisor.answer(question)
