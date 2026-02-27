import os
import re
from typing import List, Dict, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rank_bm25 import BM25Okapi

from llm import generate


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# Embeddings
# ============================================================

def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


# ============================================================
# Regex-Based Section Splitting (Structure-Aware)
# ============================================================

SECTION_HEADER_PATTERN = re.compile(
    r"""
    ^
    (?:
        \d+(\.\d+)*      # numbered headers like 1 or 1.2 or 3.4.5
        \s+
    )?
    ([A-Z][A-Z\s\-(),]{4,})   # ALL CAPS section titles
    $
    """,
    re.VERBOSE | re.MULTILINE,
)


def split_into_sections_regex(text: str) -> Dict[str, str]:
    """
    Splits protocol into sections using regex detection of headers.
    Works with numbered and ALL CAPS section headers.
    """
    matches = list(SECTION_HEADER_PATTERN.finditer(text))

    if not matches:
        return {"full_document": text}

    sections = {}
    for i, match in enumerate(matches):
        start = match.end()
        title = match.group(0).strip()

        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        sections[title] = content

    return sections


# ============================================================
# Structure-Aware Chunking
# ============================================================

def _build_documents_from_sections(
    sections: Dict[str, str],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n- ",
            "\n• ",
            "\n\n",
            "\n",
            ". ",
        ],
    )

    documents: List[Document] = []

    for title, content in sections.items():

        chunks = splitter.split_text(content)

        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "section_title": title,
                        "chunk_id": i,
                    },
                )
            )

    return documents


# ============================================================
# Hybrid RAG Index
# ============================================================

class RAGIndex:
    def __init__(self, store: FAISS, documents: List[Document]) -> None:
        self.store = store
        self.documents = documents

        # Prepare BM25 corpus
        self.corpus_texts = [doc.page_content for doc in documents]
        self.tokenized_corpus = [text.split() for text in self.corpus_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    # ----------------------------
    # Save / Load
    # ----------------------------

    def save(self, persist_dir: str) -> None:
        self.store.save_local(persist_dir)

    @classmethod
    def load(cls, persist_dir: str, embeddings) -> "RAGIndex":
        store = FAISS.load_local(
            persist_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        documents = list(store.docstore._dict.values())
        return cls(store, documents)

    # ----------------------------
    # Hybrid Retrieval
    # ----------------------------

    def hybrid_search(self, question: str, top_k: int = 8) -> List[Document]:
        # Dense retrieval
        dense_hits = self.store.similarity_search_with_score(question, k=top_k)
        dense_docs = [doc for doc, _ in dense_hits]

        # BM25 retrieval
        tokenized_query = question.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        top_bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:top_k]

        bm25_docs = [self.documents[i] for i in top_bm25_indices]

        # Reciprocal Rank Fusion (RRF) reranking
        rrf_scores = {}
        k_param = 60

        # Score dense retrieval results
        for rank, doc in enumerate(dense_docs, start=1):
            rrf_scores[doc.page_content] = rrf_scores.get(doc.page_content, 0) + 1 / (k_param + rank)

        # Score BM25 results
        for rank, doc in enumerate(bm25_docs, start=1):
            rrf_scores[doc.page_content] = rrf_scores.get(doc.page_content, 0) + 1 / (k_param + rank)

        # Get unique documents and sort by RRF score
        unique_docs = {}
        for doc in dense_docs + bm25_docs:
            if doc.page_content not in unique_docs:
                unique_docs[doc.page_content] = doc

        ranked = sorted(
            unique_docs.items(),
            key=lambda x: rrf_scores[x[0]],
            reverse=True,
        )[:top_k]

        return [doc for _, doc in ranked]

    # ----------------------------
    # Answer
    # ----------------------------

    def answer(
        self,
        question: str,
        top_k: int = 8,
        max_context_chars: int = 5000,
    ) -> str:

        docs = self.hybrid_search(question, top_k=top_k)

        if not docs:
            return "Not found in provided context.", ""

        context_blocks = []
        total_chars = 0

        for doc in docs:
            block = f"[{doc.metadata.get('section_title','')}]\n{doc.page_content}"

            if total_chars + len(block) > max_context_chars:
                break

            context_blocks.append(block)
            total_chars += len(block)

        context = "\n\n".join(context_blocks)

        prompt = f"""
You are a clinical trial protocol expert.

Answer the question STRICTLY using the context below.

Do NOT speculate, if the information is not there say it is not given in the context.
Do NOT reference sections or page numbers, just answer based on the content provided.

Context:
\"\"\"
{context}
\"\"\"

Question: {question}

Answer:
"""

        return generate(prompt).strip(), context


# ============================================================
# Builder Functions
# ============================================================

def build_rag_index_from_text(
    text: str,
    persist_dir: Optional[str] = "../data/rag_index",
    use_existing: bool = True,
) -> RAGIndex:

    embeddings = _get_embeddings()

    if persist_dir and use_existing and os.path.exists(
        os.path.join(persist_dir, "index.faiss")
    ):
        return RAGIndex.load(persist_dir, embeddings)

    sections = split_into_sections_regex(text)

    documents = _build_documents_from_sections(sections)

    store = FAISS.from_documents(documents, embeddings)

    rag_index = RAGIndex(store, documents)

    if persist_dir:
        os.makedirs(persist_dir, exist_ok=True)
        rag_index.save(persist_dir)

    return rag_index


def build_rag_index_from_pdf(
    pdf_path: str,
    persist_dir: Optional[str] = "../data/rag_index",
    use_existing: bool = True,
) -> RAGIndex:

    from parser import process_pdf

    text = process_pdf(pdf_path)

    return build_rag_index_from_text(
        text,
        persist_dir=persist_dir,
        use_existing=use_existing,
    )