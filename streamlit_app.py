import os
import sys
import tempfile
from typing import Optional

import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from parser import process_pdf
from rag import ClinicalProtocolRAG


st.set_page_config(page_title="Protocol RAG Chat", page_icon="💬", layout="wide")

st.title("Protocol Chat (RAG)")
st.caption("Upload a clinical trial protocol PDF and ask questions about it.")


def _save_temp_pdf(file_bytes: bytes) -> str:
    temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    temp_file.write(file_bytes)
    temp_file.flush()
    temp_file.close()
    return temp_file.name


@st.cache_resource(show_spinner=False)
def _build_rag(file_bytes: bytes):
    pdf_path = _save_temp_pdf(file_bytes)
    try:
        parsed_text = process_pdf(pdf_path)
        return ClinicalProtocolRAG(parsed_text)
    finally:
        try:
            os.remove(pdf_path)
        except OSError:
            pass


def _get_rag(file_bytes: Optional[bytes]):
    if not file_bytes:
        return None
    return _build_rag(file_bytes)


with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader("Upload protocol PDF", type=["pdf"])
    if uploaded_file:
        st.caption(f"Loaded: {uploaded_file.name}")

rag = None
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    with st.spinner("Processing PDF and building RAG index..."):
        rag = _get_rag(file_bytes)
else:
    st.info("📄 Upload a clinical trial protocol PDF to begin chatting.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if rag is not None:
    user_input = st.chat_input("Ask a question about the protocol")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                answer = rag.answer(user_input, conversation_history=st.session_state.messages[:-1])
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
