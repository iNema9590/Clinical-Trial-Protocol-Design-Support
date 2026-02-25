import os
from typing import Dict, List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm import generate
from parser import process_pdf
from section_splitter import split_into_sections

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def _build_documents_from_sections(
    sections: Dict[str, str],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents: List[Document] = []
    for title, content in sections.items():
        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"title": title, "chunk": i},
                )
            )
    return documents


def _has_faiss_index(persist_dir: str) -> bool:
    return os.path.exists(os.path.join(persist_dir, "index.faiss"))


def _documents_from_store(store: FAISS) -> List[Dict[str, str]]:
    documents = []
    for doc in store.docstore._dict.values():
        documents.append(
            {
                "title": doc.metadata.get("title", ""),
                "text": doc.page_content,
            }
        )
    return documents


class RAGIndex:
    def __init__(self, store: FAISS, documents: List[Dict[str, str]]) -> None:
        self.store = store
        self.documents = documents

    def save(self, persist_dir: str) -> None:
        self.store.save_local(persist_dir)

    @classmethod
    def load(cls, persist_dir: str, embeddings: HuggingFaceEmbeddings) -> "RAGIndex":
        store = FAISS.load_local(
            persist_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        documents = _documents_from_store(store)
        return cls(store, documents)

    def query(self, question: str, top_k: int = 5) -> List[Dict[str, str]]:
        hits = self.store.similarity_search_with_score(question, k=top_k)
        results = []
        for doc, score in hits:
            results.append(
                {
                    "title": doc.metadata.get("title", ""),
                    "text": doc.page_content,
                    "score": float(score),
                }
            )
        return results

    def answer(self, question: str, top_k: int = 5, max_context_chars: int = 4000) -> str:
        hits = self.query(question, top_k=top_k)
        if not hits:
            return "Not found in provided context."

        context_blocks = []
        total_chars = 0
        for hit in hits:
            block = f"[{hit['title']}]\n{hit['text']}"
            if total_chars + len(block) > max_context_chars:
                break
            context_blocks.append(block)
            total_chars += len(block)

        context = "\n\n".join(context_blocks)
        prompt = f"""
            You are a clinical trial protocol analysis expert.
            Use ONLY the context below to answer the question. If the answer is not in the context, say \"Not found in provided context.\"

            Context:
            \"\"\"
            {context}
            \"\"\"

            Question: {question}

            Answer:
            """
        return generate(prompt).strip()


def build_rag_index_from_sections(
    sections: Dict[str, str],
    chunk_size: int = 2200,
    chunk_overlap: int = 200,
    persist_dir: Optional[str] = "../data/rag_index",
    use_existing: bool = True,
) -> RAGIndex:
    embeddings = _get_embeddings()

    if persist_dir and use_existing and _has_faiss_index(persist_dir):
        return RAGIndex.load(persist_dir, embeddings)

    documents = _build_documents_from_sections(sections, chunk_size, chunk_overlap)
    store = FAISS.from_documents(documents, embeddings)
    rag_index = RAGIndex(store, _documents_from_store(store))

    if persist_dir:
        os.makedirs(persist_dir, exist_ok=True)
        rag_index.save(persist_dir)

    return rag_index


def build_rag_index_from_text(
    text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    persist_dir: Optional[str] = "../data/rag_index",
    use_existing: bool = True,
) -> RAGIndex:
    sections = split_into_sections(text)
    return build_rag_index_from_sections(
        sections,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        persist_dir=persist_dir,
        use_existing=use_existing,
    )


def build_rag_index_from_pdf(
    pdf_path: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    persist_dir: Optional[str] = "../data/rag_index",
    use_existing: bool = True,
) -> RAGIndex:
    text = process_pdf(pdf_path)
    return build_rag_index_from_text(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        persist_dir=persist_dir,
        use_existing=use_existing,
    )
