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

- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Data Schema](#data-schema)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Google Cloud credentials for Vertex AI access
- Virtual environment (venv or conda)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd Clinical-Trial-Protocol-Design-Support

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Ensure your Google credentials are available
# Place google-credentials.json in the project root
```

### Running the Application

```bash
# Start the Streamlit web interface
streamlit run streamlit_app.py

# App will open at http://localhost:8501
```

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

### Setup Steps

1. **Set Up Google Cloud Authentication**
   ```bash
   # Place credentials file in project root
   export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/google-credentials.json"
   ```

2. **Create Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python3 -c "from src.multiagent import SupervisorMultiAgent; print('✓ Setup successful')"
   ```

---

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

## 📁 Project Structure

```
Clinical-Trial-Protocol-Design-Support/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── google-credentials.json        # GCP authentication
├── streamlit_app.py              # Web UI entry point
│
├── src/                          # Core application modules
│   ├── multiagent.py             # Supervisor + agent routing (LangGraph)
│   ├── agents.py                 # Specialized extraction agents
│   ├── rag.py                    # RAG pipeline with FAISS
│   ├── llm.py                    # LLM configuration (Gemini 2.0 Flash)
│   ├── section_classifier.py     # Document section identification
│   ├── section_splitter.py       # Protocol text splitting
│   ├── structure_chunker.py      # Content chunking strategy
│   ├── structured_retriever.py   # RAG component
│   └── schemas.py                # Pydantic response schemas
│
├── data/                         # Test data and models
│   ├── synthetic_patient_data.csv        # 50 patient records (40 eligible, 10 ineligible)
│   ├── generate_synthestic_data.py       # Patient data generation script
│   ├── rag_index/                       
│   │   └── index.faiss           # FAISS vector database
│   └── [protocol PDFs]           # Test protocol documents
│
└── notebooks/                    # Jupyter notebooks
    └── pregresschecker.ipynb     # Testing & validation
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

## 📊 Data Schema

### Patient Dataset (CSV)

The eligibility checking system validates patients against 25 CSV fields:

**Demographics:**
- `PATIENT_ID`, `AGE`, `BMI`

**Laboratory Values:**
- `ALT` (liver enzyme), `ULN` (upper limit normal)

**Test Results:**
- `PCR_RESULT` (Positive/Negative)

**Evaluable Criteria Flags (Boolean/Categorical):**
- `SEVERE_ALLERGY_HISTORY` (True/False)
- `IMMUNOSUPPRESSIVE_THERAPY_6M` (True/False)
- `PREGNANT` (True/False)
- `BODY_TEMPERATURE` (numeric, ≤37.5°C normal)
- `RECENT_VACCINE_30D`, `RECENT_BLOOD_DONATION_30D` (True/False)
- `SARS_COV2_RISK_LEVEL` (Low/High)
- `MEDICAL_STABILITY_STATUS` (Stable/Unstable)
- `COGNITIVE_COMPLIANCE_CAPABLE` (True/False)
- `USING_CONTRACEPTION` (True/False)
- `CONSENTED` (True/False)
- `GUILLAIN_BARRE_HISTORY`, `IMMUNODEFICIENCY_CONDITION` (True/False)
- `MALIGNANCY_HISTORY`, `BLEEDING_DISORDER_HISTORY` (True/False)
- `SEVERE_COMORBIDITY` (True/False)
- `INVESTIGATIONAL_SARS_COV2_DRUG` (True/False)
- `RECENT_IMMUNOGLOBULIN_3M` (True/False)
- `STUDY_STAFF_INVOLVEMENT` (True/False)

### Extraction Schemas

**EligibilityCriteria**
```python
{
  "evaluable_criteria": [
    {
      "criterion": "Positive PCR test result",
      "field_name": "PCR_RESULT",
      "operator": "==",
      "values": ["Positive"],
      "criterion_type": "exclusion"
    }
  ],
  "non_evaluable_criteria": [
    {
      "criterion": "No known contraindications",
      "reason": "Not machine-evaluable; requires medical judgment"
    }
  ]
}
```

**EligibilityCheckResult**
```python
{
  "total_patients": 50,
  "eligible_count": 40,
  "non_eligible_count": 10,
  "evaluated_rules": [
    {
      "rule": "PCR_RESULT == Negative",
      "evaluation_type": "evaluable"
    }
  ],
  "unevaluated_criteria": [
    {
      "criterion": "No known contraindications",
      "reason": "Not machine-evaluable"
    }
  ],
  "non_eligible_patients": [
    {
      "patient_id": "41",
      "violations": [
        {
          "field": "PCR_RESULT",
          "actual_value": "Positive",
          "rule": "PCR_RESULT == Negative"
        }
      ],
      "record": {...}
    }
  ]
}
```

---

## 🔧 Troubleshooting

### Issue: `google.api_core.exceptions.AuthenticationError`
**Solution:** Ensure `GOOGLE_APPLICATION_CREDENTIALS` is set and points to valid credentials file.

### Issue: FAISS Index Not Found
**Solution:** Run protocol extraction once to initialize RAG index automatically.

### Issue: Supervisor Always Routes to RAG
**Solution:** The supervisor uses keyword matching with LLM fallback. Ensure query contains relevant keywords (see routing table).

### Issue: Patient Eligibility Check Returns Empty
**Solution:** 
1. First extract eligibility criteria with: "What are the inclusion and exclusion criteria?"
2. Then validate patients with: "Check patients against these criteria"

### Issue: Memory Issues with Large PDFs
**Solution:** Adjust chunk size in [src/structure_chunker.py](src/structure_chunker.py) or switch to `faiss-gpu` for vector operations.

---

## 📚 Example Queries

The system understands natural language queries across multiple domains:

```
# Objectives
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

## 🤝 Contributing

To add new agents or routes:

1. **Create extraction function** in [src/agents.py](src/agents.py)
2. **Define Pydantic schema** in [src/schemas.py](src/schemas.py)
3. **Add agent node** to `DocumentMultiAgentCore` in [src/multiagent.py](src/multiagent.py)
4. **Update supervisor routing** with new keyword triggers

---

## 📄 License

[Add your license here]

---

## 📧 Support

For issues or questions, please open a GitHub issue or contact the development team.