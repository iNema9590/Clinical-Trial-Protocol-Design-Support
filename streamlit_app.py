import io
import os
import sys
import tempfile
from typing import Optional, Tuple

import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from rag import build_rag_index_from_pdf


st.set_page_config(page_title="Protocol RAG Chat", page_icon="💬", layout="wide")

st.title("Protocol Chat (RAG)")
st.caption("Upload a protocol PDF and chat with it using RAG.")


def _save_temp_pdf(file_bytes: bytes) -> str:
    temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    temp_file.write(file_bytes)
    temp_file.flush()
    temp_file.close()
    return temp_file.name


@st.cache_resource(show_spinner=False)
def _build_index(file_bytes: bytes):
    pdf_path = _save_temp_pdf(file_bytes)
    try:
        return build_rag_index_from_pdf(pdf_path, persist_dir=None, use_existing=False)
    finally:
        try:
            os.remove(pdf_path)
        except OSError:
            pass


def _get_index(file_bytes: Optional[bytes]):
    if not file_bytes:
        return None
    return _build_index(file_bytes)


with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader("Upload protocol PDF", type=["pdf"])
    st.markdown("---")
    top_k = st.slider("Top K chunks", min_value=2, max_value=12, value=6, step=1)
    show_context = st.toggle("Show retrieved context", value=False)

index = None
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    with st.spinner("Building RAG index..."):
        index = _get_index(file_bytes)
else:
    st.info("Upload a PDF to begin.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if index is not None:
    user_input = st.chat_input("Ask a question about the protocol")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, context = index.answer(user_input, top_k=top_k)
            st.markdown(answer)
            if show_context and context:
                with st.expander("Retrieved context"):
                    st.text(context)

        st.session_state.messages.append({"role": "assistant", "content": answer})
