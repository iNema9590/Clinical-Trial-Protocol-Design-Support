import re
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from structure_chunker import build_structured_chunks
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi

from llm import generate


# -----------------------------
# TOKENIZER (same as embedding model)
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# TABLE DETECTION
# -----------------------------
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


# -----------------------------
# BUILD DOCUMENTS
# -----------------------------
def build_qa_documents(structured_chunks, max_tokens=800, overlap=150):

    documents = []

    for chunk in structured_chunks:
        blocks = split_text_and_tables(chunk.content)

        for block in blocks:

            # Tables stay atomic
            if block["type"] == "table":
                content = f"""
Section Path: {chunk.full_title}
Section Number: {chunk.section_id}

{block["content"]}
"""
                documents.append(
                    Document(
                        page_content=content.strip(),
                        metadata={
                            "section_id": chunk.section_id,
                            "full_title": chunk.full_title,
                            "is_table": True
                        }
                    )
                )
                continue

            # Text blocks use sliding window
            tokens = tokenizer.encode(block["content"])

            if len(tokens) <= max_tokens:
                windows = [tokens]
            else:
                windows = []
                start = 0
                while start < len(tokens):
                    end = start + max_tokens
                    windows.append(tokens[start:end])
                    start += max_tokens - overlap

            for window_tokens in windows:
                window_text = tokenizer.decode(window_tokens)

                content = f"""
Section Path: {chunk.full_title}
Section Number: {chunk.section_id}

{window_text}
"""

                documents.append(
                    Document(
                        page_content=content.strip(),
                        metadata={
                            "section_id": chunk.section_id,
                            "full_title": chunk.full_title,
                            "is_table": False
                        }
                    )
                )

    return documents


# -----------------------------
# HYBRID RAG CLASS
# -----------------------------
class ClinicalProtocolRAG:

    def __init__(self, raw_text: str):

        print("Building structured chunks...")
        structured_chunks = build_structured_chunks(raw_text)

        print("Building QA documents...")
        self.documents = build_qa_documents(structured_chunks)

        print("Building dense FAISS index...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True}
        )

        self.vectorstore = FAISS.from_documents(
            self.documents,
            self.embeddings
        )

        print("Building BM25 index...")
        tokenized_corpus = [
            doc.page_content.lower().split()
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

        print("Hybrid RAG ready.")

    # --------------------------------
    # METADATA FILTERING
    # --------------------------------
    def _apply_metadata_filters(
        self,
        docs: List[Document],
        filters: Optional[Dict]
    ) -> List[Document]:

        if not filters:
            return docs

        filtered_docs = []

        for doc in docs:
            match = True
            for key, value in filters.items():
                if doc.metadata.get(key) != value:
                    match = False
                    break
            if match:
                filtered_docs.append(doc)

        return filtered_docs

    # --------------------------------
    # HYBRID RETRIEVAL
    # --------------------------------
    def retrieve(
        self,
        question: str,
        k: int = 6,
        filters: Optional[Dict] = None
    ) -> List[Document]:

        # ----- Dense Retrieval -----
        dense_docs = self.vectorstore.similarity_search_with_score(
            question,
            k=len(self.documents)
        )

        # Rank by dense retrieval (lower distance = better rank)
        dense_ranked = sorted(dense_docs, key=lambda x: x[1])
        dense_ranks = {
            doc.page_content: rank + 1  # 1-indexed ranks
            for rank, (doc, _) in enumerate(dense_ranked)
        }

        # ----- BM25 Retrieval -----
        tokenized_query = question.lower().split()
        bm25_raw_scores = self.bm25.get_scores(tokenized_query)

        # Rank by BM25 scores (higher = better)
        bm25_ranked_indices = sorted(
            range(len(bm25_raw_scores)), 
            key=lambda i: bm25_raw_scores[i],
            reverse=True
        )
        bm25_ranks = {
            self.documents[idx].page_content: rank + 1  # 1-indexed ranks
            for rank, idx in enumerate(bm25_ranked_indices)
        }

        # ----- Reciprocal Rank Fusion (RRF) -----
        k_constant = 60  # Standard RRF constant
        rrf_scores = {}

        for doc in self.documents:
            content = doc.page_content
            
            dense_rank = dense_ranks.get(content, len(self.documents) + 1)
            bm25_rank = bm25_ranks.get(content, len(self.documents) + 1)
            
            # RRF formula: sum of 1/(k + rank) for each retrieval method
            rrf_scores[content] = (
                1 / (k_constant + dense_rank) +
                1 / (k_constant + bm25_rank)
            )

        # Rank by RRF scores
        ranked_contents = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        ranked_docs = [
            next(doc for doc in self.documents if doc.page_content == content)
            for content, _ in ranked_contents
        ]

        # Apply metadata filtering
        ranked_docs = self._apply_metadata_filters(
            ranked_docs,
            filters
        )

        return ranked_docs[:k]

    # --------------------------------
    # ANSWER GENERATION
    # --------------------------------
    def answer(
        self,
        question: str,
        k: int = 6,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        filters: Optional[Dict] = None
    ):

        docs = self.retrieve(
            question,
            k=k,
            filters=filters
        )

        context = "\n\n".join([doc.page_content for doc in docs])

        history_str = ""
        if conversation_history:
            history_lines = []
            for msg in conversation_history:
                role = msg.get("role", "unknown").capitalize()
                content = msg.get("content", "")
                history_lines.append(f"{role}: {content}")
            history_str = "\n".join(history_lines) + "\n\n"

        prompt = f"""You are a clinical trial protocol expert.

Use ONLY the provided context to answer the question. Answer in a thorough and detailed manner, including relevant specifics and criteria from the protocol.
If the answer is not explicitly stated, say:
"Not specified in the protocol."

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




# import re
# from typing import List, Dict, Optional
# from dataclasses import dataclass

# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document
# from structure_chunker import build_structured_chunks
# from transformers import AutoTokenizer

# from llm import generate


# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# TABLE_BLOCK_PATTERN = re.compile(
#     r"(\|.+?\|\n\|[-:\s|]+\|(?:\n\|.*?\|)+)",
#     re.DOTALL
# )


# def split_text_and_tables(text):
#     parts = TABLE_BLOCK_PATTERN.split(text)
#     blocks = []

#     for part in parts:
#         if not part.strip():
#             continue

#         if TABLE_BLOCK_PATTERN.match(part):
#             blocks.append({"type": "table", "content": part.strip()})
#         else:
#             blocks.append({"type": "text", "content": part.strip()})

#     return blocks


# def build_qa_documents(structured_chunks, max_tokens=800, overlap=150):

#     documents = []

#     for chunk in structured_chunks:
#         blocks = split_text_and_tables(chunk.content)

#         for block in blocks:

#             # TABLE = atomic
#             if block["type"] == "table":
#                 content = f"""
# Section Path: {chunk.full_title}
# Section Number: {chunk.section_id}

# {block["content"]}
# """
#                 documents.append(
#                     Document(
#                         page_content=content,
#                         metadata={
#                             "section_id": chunk.section_id,
#                             "full_title": chunk.full_title,
#                             "is_table": True
#                         }
#                     )
#                 )
#                 continue

#             # TEXT = adaptive sliding window
#             tokens = tokenizer.encode(block["content"])

#             if len(tokens) <= max_tokens:
#                 window_text = block["content"]

#                 content = f"""
# Section Path: {chunk.full_title}
# Section Number: {chunk.section_id}

# {window_text}
# """

#                 documents.append(
#                     Document(
#                         page_content=content,
#                         metadata={
#                             "section_id": chunk.section_id,
#                             "full_title": chunk.full_title,
#                             "is_table": False
#                         }
#                     )
#                 )
#             else:
#                 start = 0
#                 while start < len(tokens):
#                     end = start + max_tokens
#                     window_tokens = tokens[start:end]
#                     window_text = tokenizer.decode(window_tokens)

#                     content = f"""
# Section Path: {chunk.full_title}
# Section Number: {chunk.section_id}

# {window_text}
# """

#                     documents.append(
#                         Document(
#                             page_content=content,
#                             metadata={
#                                 "section_id": chunk.section_id,
#                                 "full_title": chunk.full_title,
#                                 "is_table": False
#                             }
#                         )
#                     )

#                     start += max_tokens - overlap

#     return documents

# def build_faiss_index(documents):

#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#     encode_kwargs={"normalize_embeddings": True}
#     )

#     vectorstore = FAISS.from_documents(
#         documents,
#         embeddings
#     )

#     return vectorstore


# class ClinicalProtocolRAG:

#     def __init__(self, raw_text: str):

#         print("Building structured chunks...")
#         structured_chunks = build_structured_chunks(raw_text)

#         print("Building QA documents...")
#         documents = build_qa_documents(structured_chunks)

#         print("Building FAISS index...")
#         self.vectorstore = build_faiss_index(documents)

#     def retrieve(self, question: str, k: int = 6):

#         retriever = self.vectorstore.as_retriever(
#             search_kwargs={"k": k}
#         )

#         docs = retriever.invoke(question)

#         return docs

#     def answer(self, question: str, k: int = 6, conversation_history: Optional[List[Dict[str, str]]] = None):

#         docs = self.retrieve(question, k)

#         context = "\n\n".join([doc.page_content for doc in docs])

#         # Build conversation history string
#         history_str = ""
#         if conversation_history:
#             history_lines = []
#             for msg in conversation_history:
#                 role = msg.get("role", "unknown").capitalize()
#                 content = msg.get("content", "")
#                 history_lines.append(f"{role}: {content}")
#             history_str = "\n".join(history_lines) + "\n\n"

#         prompt = f"""You are a clinical trial protocol expert. 
#         You analyze the provided sections of a clinical trial protocol to answer questions about the study design, objectives, eligibility criteria, assessments, and other details. 
#         Use ONLY the provided context to answer the question. 
#         If the answer is not explicitly stated in the context, say: "Not specified in the protocol." 
#         Provide a thorough and detailed explanation, including relevant specifics and criteria from the protocol.


# ==================== CONVERSATION HISTORY ====================
# {history_str}
# ==================== PROTOCOL CONTEXT ====================

# {context}

# ===========================================================

# Question: {question}

# Answer:
# """

#         response = generate(prompt)

#         return response
#     # Be concise and clear in your response, as if explaining to a non-expert.