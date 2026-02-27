import re
from typing import List, Dict, Optional
from dataclasses import dataclass

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from structure_chunker import build_structured_chunks
from transformers import AutoTokenizer

from llm import generate


tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

TABLE_BLOCK_PATTERN = re.compile(
    r"(\|.+?\|\n\|[-:\s|]+\|(?:\n\|.*?\|)+)",
    re.DOTALL
)


def split_text_and_tables(text):
    parts = TABLE_BLOCK_PATTERN.split(text)
    blocks = []

    for part in parts:
        if not part.strip():
            continue

        if TABLE_BLOCK_PATTERN.match(part):
            blocks.append({"type": "table", "content": part.strip()})
        else:
            blocks.append({"type": "text", "content": part.strip()})

    return blocks


def build_qa_documents(structured_chunks, max_tokens=800, overlap=150):

    documents = []

    for chunk in structured_chunks:
        blocks = split_text_and_tables(chunk.content)

        for block in blocks:

            # TABLE = atomic
            if block["type"] == "table":
                content = f"""
Section Path: {chunk.full_title}
Section Number: {chunk.section_id}

{block["content"]}
"""
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "section_id": chunk.section_id,
                            "full_title": chunk.full_title,
                            "is_table": True
                        }
                    )
                )
                continue

            # TEXT = adaptive sliding window
            tokens = tokenizer.encode(block["content"])

            if len(tokens) <= max_tokens:
                window_text = block["content"]

                content = f"""
Section Path: {chunk.full_title}
Section Number: {chunk.section_id}

{window_text}
"""

                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "section_id": chunk.section_id,
                            "full_title": chunk.full_title,
                            "is_table": False
                        }
                    )
                )
            else:
                start = 0
                while start < len(tokens):
                    end = start + max_tokens
                    window_tokens = tokens[start:end]
                    window_text = tokenizer.decode(window_tokens)

                    content = f"""
Section Path: {chunk.full_title}
Section Number: {chunk.section_id}

{window_text}
"""

                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "section_id": chunk.section_id,
                                "full_title": chunk.full_title,
                                "is_table": False
                            }
                        )
                    )

                    start += max_tokens - overlap

    return documents

def build_faiss_index(documents):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )

    return vectorstore


class ClinicalProtocolRAG:

    def __init__(self, raw_text: str):

        print("Building structured chunks...")
        structured_chunks = build_structured_chunks(raw_text)

        print("Building QA documents...")
        documents = build_qa_documents(structured_chunks)

        print("Building FAISS index...")
        self.vectorstore = build_faiss_index(documents)

    def retrieve(self, question: str, k: int = 6):

        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

        docs = retriever.invoke(question)

        return docs

    def answer(self, question: str, k: int = 6, conversation_history: Optional[List[Dict[str, str]]] = None):

        docs = self.retrieve(question, k)

        context = "\n\n".join([doc.page_content for doc in docs])

        # Build conversation history string
        history_str = ""
        if conversation_history:
            history_lines = []
            for msg in conversation_history:
                role = msg.get("role", "unknown").capitalize()
                content = msg.get("content", "")
                history_lines.append(f"{role}: {content}")
            history_str = "\n".join(history_lines) + "\n\n"

        prompt = f"""You are a clinical trial protocol expert.

Answer the question using ONLY the provided protocol context.
If the answer is not explicitly stated, say: "Not specified in the protocol."
Do not reference section numbers or anything else.
Be gentle and concise in your answer, as if you were talking to a non-expert.

==================== CONVERSATION HISTORY ====================
{history_str}
==================== PROTOCOL CONTEXT ====================

{context}

===========================================================

Question: {question}

Answer:
"""

        response = generate(prompt)

        return response