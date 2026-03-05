import os
import sys
import tempfile
import json
from typing import Optional

import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from parser import process_pdf
from multiagent import SupervisorMultiAgent
from structure_chunker import build_structured_chunks


st.set_page_config(page_title="Protocol Intelligence Assistant", page_icon="🔬", layout="wide")

st.title("🔬 Clinical Trial Protocol Intelligence Assistant")
st.caption("Upload a clinical trial protocol PDF and ask questions. Queries are intelligently routed to specialized agents.")


def _save_temp_pdf(file_bytes: bytes) -> str:
    temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    temp_file.write(file_bytes)
    temp_file.flush()
    temp_file.close()
    return temp_file.name


@st.cache_resource(show_spinner=False)
def _build_systems(file_bytes: bytes):
    """Build MultiAgent system from PDF (includes RAG internally)."""
    pdf_path = _save_temp_pdf(file_bytes)
    try:
        # Parse PDF
        parsed_text = process_pdf(pdf_path)
        
        # Build sections for multiagent
        chunks = build_structured_chunks(parsed_text)
        sections_dict = {chunk.full_title: chunk.content for chunk in chunks}
        
        # Build multiagent system (includes RAG internally)
        multiagent = SupervisorMultiAgent(
            sections=sections_dict,
            parsed_text=parsed_text,
            rag_persist_dir="data/rag_index",
            patient_data_path="data/synthetic_patient_data.csv"
        )
        
        return {
            "multiagent": multiagent,
            "sections_count": len(sections_dict),
            "parsed_text_length": len(parsed_text)
        }
    finally:
        try:
            os.remove(pdf_path)
        except OSError:
            pass


def _get_systems(file_bytes: Optional[bytes]):
    """Get multiagent system."""
    if not file_bytes:
        return None
    return _build_systems(file_bytes)


def _build_conversation_history(
    messages: list[dict],
    max_turn_messages: int = 10,
) -> list[dict[str, str]]:
    """Build a compact role/content history for memory-aware answering."""
    history: list[dict[str, str]] = []
    for msg in messages[-max_turn_messages:]:
        role = msg.get("role")
        content = msg.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            history.append({"role": role, "content": content})
    return history


with st.sidebar:
    st.header("📄 Document")
    uploaded_file = st.file_uploader("Upload protocol PDF", type=["pdf"])
    
    if uploaded_file:
        st.success(f"✓ Loaded: {uploaded_file.name}")
    
    st.divider()
    
    # Agent capabilities info
    with st.expander("🤖 Agent Capabilities", expanded=False):
        st.markdown("""
        **Intelligent Routing to Specialized Agents:**
        
        - 🎯 **Objectives** - Study objectives & endpoints
        - ✅ **Eligibility** - Inclusion/exclusion criteria
        - 🔍 **Eligibility Check** - Validate patient datasets
        - 📅 **SoA** - Schedule of Activities
        - 🏥 **Visit Definitions** - Visit timing & windows
        - 📊 **Key Assessments** - Assessment procedures
        - 🔎 **RAG** - General queries (fallback)
        
        The system automatically routes your question to the most appropriate agent.
        """)
    
    with st.expander("💡 Example Questions", expanded=False):
        st.markdown("""
        **Objectives:**
        - "What are the primary objectives?"
        - "List all endpoints"
        
        **Eligibility:**
        - "What are the inclusion criteria?"
        - "Show exclusion criteria"
        
        **Eligibility Check:**
        - "Perform data validation with the eligibility criteria"
        - "Check which patients are not eligible"
        
        **Schedule:**
        - "Show the schedule of activities"
        - "When does screening occur?"
        
        **Assessments:**
        - "What are the key assessments?"
        """)

systems = None
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    with st.spinner("🔄 Processing PDF and building intelligence system..."):
        systems = _get_systems(file_bytes)
        if systems:
            st.sidebar.info(f"📊 Parsed {systems['sections_count']} sections ({systems['parsed_text_length']:,} chars)")
else:
    st.info("📄 **Get Started:** Upload a clinical trial protocol PDF in the sidebar to begin.")
    
    # Show welcome info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 🎯 Intelligent Routing
        Questions are automatically routed to specialized agents for precise extraction.
        """)
    with col2:
        st.markdown("""
        ### 📊 Structured Data
        Extract objectives, eligibility, schedules, and assessments in structured format.
        """)
    with col3:
        st.markdown("""
        ### 🔍 Data Validation
        Validate patient datasets against protocol eligibility criteria.
        """)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear chat button (only show when there are messages)
if systems is not None and len(st.session_state.messages) > 0:
    if st.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show routing info if available (for multiagent mode)
        if message["role"] == "assistant" and "metadata" in message:
            with st.expander("🔍 Routing Details", expanded=False):
                metadata = message["metadata"]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Route", metadata.get("route", "unknown"))
                with col2:
                    st.metric("Source", metadata.get("source", "unknown"))
                if "routing_info" in metadata:
                    routing_info = metadata["routing_info"]
                    if "reason" in routing_info:
                        st.caption(f"**Reason:** {routing_info['reason']}")

if systems is not None:
    user_input = st.chat_input("Ask a question about the protocol")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Processing your question..."):
                conversation_history = _build_conversation_history(
                    st.session_state.messages,
                    max_turn_messages=10,
                )

                # Use multiagent system
                response_json = systems["multiagent"].answer(
                    user_input,
                    conversation_history=conversation_history,
                )
                try:
                    response_data = json.loads(response_json)
                    route = response_data.get("route", "unknown")
                    
                    # Display routing badge
                    route_emoji = {
                        "objectives and endpoints": "🎯",
                        "eligibility": "✅",
                        "eligibility check": "🔍",
                        "schedule of activities": "📅",
                        "visit definitions": "🏥",
                        "key assessments": "📊",
                        "rag": "🔎"
                    }
                    st.caption(f"{route_emoji.get(route, '❓')} Routed to: **{route}**")
                    
                    # Handle eligibility check specially
                    if route == "eligibility check":
                        # Display eligibility check results
                        total = response_data.get("total_patients", 0)
                        non_eligible_count = response_data.get("non_eligible_count", 0)
                        non_eligible = response_data.get("non_eligible_patients", [])
                        evaluated = response_data.get("evaluated_rules", [])
                        unevaluated = response_data.get("unevaluated_criteria", [])
                        
                        st.markdown(f"### Eligibility Validation Results")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Patients", total)
                        with col2:
                            st.metric("Non-Eligible", non_eligible_count)
                        with col3:
                            st.metric("Eligible", total - non_eligible_count)
                        
                        if evaluated:
                            with st.expander("📋 Evaluated Rules", expanded=True):
                                for rule in evaluated:
                                    st.markdown(f"- **{rule['type'].title()}:** {rule['criterion']}")
                                    st.caption(f"   → Parsed as: `{rule['parsed_rule']}`")
                        
                        if unevaluated:
                            with st.expander("⚠️ Unevaluated Criteria", expanded=False):
                                for item in unevaluated:
                                    st.markdown(f"- **{item['type'].title()}:** {item['criterion']}")
                                    st.caption(f"   → {item['reason']}")
                        
                        if non_eligible:
                            with st.expander(f"❌ Non-Eligible Patients ({len(non_eligible)})", expanded=True):
                                for patient in non_eligible:
                                    st.markdown(f"**Patient {patient['patient_id']}:**")
                                    for reason in patient['reasons']:
                                        st.markdown(f"  - {reason}")
                                    st.caption(f"  Record: {patient['patient_record']}")
                                    st.divider()
                        
                        answer_text = f"Validation complete: {non_eligible_count}/{total} patients are non-eligible."
                    
                    else:
                        # Handle other routes
                        answer = response_data.get("answer", "No answer available")
                        
                        # Handle different response formats
                        if isinstance(answer, dict):
                            # Structured extraction response
                            st.json(answer)
                            answer_text = f"**Extracted Information:**\n\n```json\n{json.dumps(answer, indent=2)}\n```"
                        elif isinstance(answer, str):
                            answer_text = answer
                            st.markdown(answer_text)
                        else:
                            answer_text = str(answer)
                            st.markdown(answer_text)
                    
                    # Store metadata
                    metadata = {
                        "route": route,
                        "source": response_data.get("source", "unknown"),
                        "routing_info": response_data.get("routing_info", {}),
                    }
                    
                except json.JSONDecodeError:
                    answer_text = response_json
                    metadata = {"route": "error", "source": "multiagent"}
                    st.markdown(answer_text)

        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer_text,
            "metadata": metadata
        })
