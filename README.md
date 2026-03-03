# Clinical Trial Protocol Design Support System

An intelligent multi-agent system for automated analysis of clinical trial protocols using LangGraph, LLM-based routing, and Retrieval-Augmented Generation (RAG). The system extracts key information from protocol documents, validates patient eligibility against real datasets, and provides structured JSON responses.

---

## 🎯 Features

✅ **Intelligent Multi-Agent Routing** — LLM-based supervisor routes queries to specialized agents  
✅ **7 Specialized Agents** — Objectives, eligibility extraction, schedule, visits, assessments, eligibility checking, and RAG fallback  
✅ **Patient Eligibility Validation** — Evaluates patient datasets against extracted eligibility criteria  
✅ **Dynamic Rule Evaluation** — Automatically detects field types (boolean, numeric, categorical) and applies appropriate operators  
✅ **RAG Integration** — FAISS-powered semantic search for comprehensive protocol analysis  
✅ **Streamlit Web UI** — Interactive interface with rich eligibility dashboards  
✅ **Structured Output** — Pydantic-validated JSON responses with detailed parsing information  

---

## 📋 Table of Contents

- [System Architecture](#system-architecture)
- [Usage](#usage)
- [API Reference](#api-reference)
---

## 🏗️ System Architecture

### Overall Flow

```
┌─────────────────────────────┐
│  Protocol PDF Upload (UI)   │
└────────────┬────────────────┘
             │
    ┌────────▼────────────────────────────────┐
    │  Supervisor Agent (Router)              │
    │  - Analyzes user query                  │
    │  - Decides optimal agent route          │
    │  - Keyword override for validation      │
    └────────┬─────────────────────────────────┘
             │
    ┌────────┴─────────────────────────────────┐
    │                                          │
    ├─► Objectives Agent ────► Extract study goals & endpoints
    │
    ├─► Eligibility Agent ───► Parse inclusion/exclusion criteria
    │
    ├─► Eligibility Check ───► Validate patients against rules
    │
    ├─► SOA Agent ──────────► Extract schedule of activities
    │
    ├─► Visit Definitions ──► Parse visit windows & timings
    │
    ├─► Key Assessments ────► Extract safety & efficacy assessments
    │
    └─► RAG Agent ─────────► General protocol knowledge retrieval
             │
             └────────────────────┐
                                  │
                    ┌─────────────▼──────────────┐
                    │ Format JSON Response       │
                    │ - Add routing metadata     │
                    │ - Include source sections  │
                    │ - Serialize results        │
                    └────────────────────────────┘
```

### Agent Routing Table

| Route | Keywords | Purpose | Output |
|-------|----------|---------|--------|
| **objectives** | "objective", "endpoint", "goal", "aim" | Extract primary/secondary objectives | ObjectivesResponse |
| **eligibility** | "inclusion", "exclusion", "eligible", "criteria" | Extract inclusion/exclusion criteria | EligibilityCriteria |
| **eligibility_check** | "validate", "check patient", "validate patients" | Check patients against rules | EligibilityCheckResult |
| **soa** | "schedule", "activities", "procedures", "timeline" | Extract schedule of activities | ScheduleOfActivities |
| **visit_definitions** | "visit", "window", "screening", "timing" | Extract visit details & timing | VisitDefinitions |
| **key_assessments** | "assessment", "safety", "evaluation", "measure" | Extract assessments & procedures | KeyAssessments |
| **rag** | *Any unmatched query* | Fallback for general questions | RAGResponse |

---

## 💻 Installation

### Dependencies

```
python>=3.9
langchain-core>=0.1.0
langgraph>=0.0.50
langchain-google-vertexai>=0.0.30
faiss-cpu>=1.7.4
pandas>=2.0.0
pydantic>=2.0.0
streamlit>=1.28.0
```

## 📖 Usage

### Web Application (Streamlit)

```bash
streamlit run streamlit_app.py
```

**Features:**
- Upload protocol PDF via file uploader
- Ask questions in natural language
- View intelligent routing badges (🎯✅🔍📅🏥📊🔎)
- See detailed eligibility validation results with metrics
- Review unevaluated criteria requiring manual review
- View non-eligible patient details with violation reasons

### Python API

```python
from src.multiagent import SupervisorMultiAgent
from src.section_splitter import SectionSplitter
import PyPDF2

# Parse PDF
pdf_path = "protocol.pdf"
with open(pdf_path, 'rb') as f:
    reader = PyPDF2.PdfReader(f)
    text = "".join(page.extract_text() for page in reader.pages)

# Split into sections
splitter = SectionSplitter()
sections = splitter.split_protocol(text)

# Create supervisor
supervisor = SupervisorMultiAgent(sections, text)

# Query
result = supervisor.answer("What are the inclusion criteria?")
print(result)
```

---

## 🔌 API Reference

### SupervisorMultiAgent

Main orchestrator class for intelligent routing and multi-agent execution.

#### Initialization
```python
SupervisorMultiAgent(
    sections: dict,           # Parsed protocol sections
    parsed_text: str,         # Full protocol text
    llm=None                  # Optional LLM override
)
```

#### Methods

**`answer(query: str) -> dict`**
- Routes query to appropriate agent
- Returns formatted JSON response
- Includes routing metadata and source sections

```python
response = supervisor.answer("What are the exclusion criteria?")
# Returns:
# {
#   "question": "What are the exclusion criteria?",
#   "route": "eligibility",
#   "routing_info": {"reason": "Query contains exclusion keyword"},
#   "source": "eligibility_agent",
#   "answer": {"evaluable_criteria": [...], "non_evaluable_criteria": [...]},
#   "selected_sections": ["Eligibility Criteria"]
# }
```

### Eligibility Checking

```python
# After extracting eligibility criteria
response = supervisor.answer("validate patients against the criteria")
# Returns eligibility check results with:
# - Total patient count
# - Eligible count
# - Non-eligible count
# - Detailed violation reasons per patient
```

---

## ⚙️ Configuration

### Environment Variables

```bash
export GOOGLE_APPLICATION_CREDENTIALS="./google-credentials.json"
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

### LLM Configuration

Edit [src/llm.py](src/llm.py) to customize:
- Model: `gemini-2.0-flash-001`
- Temperature: `0.0` (for routing), varies per task
- Region: `us-central1`

### RAG Configuration

Edit [src/rag.py](src/rag.py) to adjust:
- Vector database: FAISS
- Chunk size: 1000 tokens
- Overlap: 200 tokens
- Similarity threshold: 0.5

---

## 📚 Example Queries

The system understands natural language queries across multiple domains:

```
# Objectives and Endpoints
"What are the study objectives?"
"What endpoints will be measured?"
"What are the goals of this trial?"

# Eligibility
"Who can participate in this study?"
"What are the exclusion criteria?"
"List all inclusion criteria"

# Patient Validation
"Validate our patient dataset"
"Check which patients are eligible"
"How many patients meet the criteria?"

# Schedule
"What is the schedule of activities?"
"What procedures are done at each visit?"
"How long is the study?"

# Visit Details
"What is the screening window?"
"When are patients seen?"
"Describe the week 12 visit"

# Assessments
"What safety assessments are required?"
"List all study procedures"
"What endpoints will be evaluated?"
```

---

## 🎓 Architecture Deep Dive

### LangGraph State Machine

The system uses LangGraph to orchestrate a directed acyclic graph (DAG):

1. **Supervisor Node** → Analyzes query, selects route
2. **Agent Nodes** (7 variants) → Execute specialized task
3. **Final Node** → Format and return response

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)

# Add nodes
graph.add_node("supervisor", supervisor_node)
graph.add_node("objectives", objectives_agent)
graph.add_node("eligibility", eligibility_agent)
# ... (5 more agent nodes)
graph.add_node("final", final_node)

# Add conditional routing
graph.add_conditional_edges("supervisor", route_decision)

# All agents → final node
for agent in ["objectives", "eligibility", ...]:
    graph.add_edge(agent, "final")

graph.add_edge("final", END)
```

### Dynamic Rule Evaluation

The `eligibility_check_agent()` automatically detects field types from actual CSV data:

```python
def _evaluate_rule(self, field_name, operator, values, data):
    col = data[field_name]
    unique_vals = col.astype(str).str.strip().unique()
    
    # Detect field type
    if set(unique_vals).issubset({"True", "False"}):
        # Boolean field
        return self._evaluate_boolean(col, operator, values)
    elif len(unique_vals) <= 10:
        # Categorical field
        return self._evaluate_categorical(col, operator, values)
    else:
        # Numeric field
        return self._evaluate_numeric(col, operator, values)
```

No hardcoded field lists—system adapts to any dataset schema.

---