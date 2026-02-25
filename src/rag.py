import re
from typing import Dict, List

import torch

from llm import generate
from parser import process_pdf
from section_classifier import embed
from section_splitter import split_into_sections


def _chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    chunks = []
    start = 0
    length = len(cleaned)

    while start < length:
        end = min(length, start + max_chars)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = max(0, end - overlap)

    return chunks


def _embed_in_batches(texts: List[str], batch_size: int) -> torch.Tensor:
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings.append(embed(batch))
    return torch.cat(embeddings, dim=0) if embeddings else torch.empty((0, 0))


class RAGIndex:
    def __init__(self, documents: List[Dict[str, str]], embeddings: torch.Tensor) -> None:
        self.documents = documents
        self.embeddings = embeddings

    def query(self, question: str, top_k: int = 5) -> List[Dict[str, str]]:
        if not self.documents:
            return []

        query_embedding = embed([question])[0]
        scores = torch.matmul(self.embeddings, query_embedding)
        top_scores, top_indices = torch.topk(scores, k=min(top_k, len(self.documents)))

        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            doc = self.documents[idx]
            results.append(
                {
                    "title": doc["title"],
                    "text": doc["text"],
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
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    batch_size: int = 16,
) -> RAGIndex:
    documents = []
    for title, content in sections.items():
        chunks = _chunk_text(content, max_chars=chunk_size, overlap=chunk_overlap)
        for i, chunk in enumerate(chunks):
            documents.append({"id": f"{title}::chunk{i}", "title": title, "text": chunk})

    embeddings = _embed_in_batches([doc["text"] for doc in documents], batch_size)
    return RAGIndex(documents, embeddings)


def build_rag_index_from_text(
    text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    batch_size: int = 16,
) -> RAGIndex:
    sections = split_into_sections(text)
    return build_rag_index_from_sections(
        sections,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        batch_size=batch_size,
    )


def build_rag_index_from_pdf(
    pdf_path: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    batch_size: int = 16,
) -> RAGIndex:
    text = process_pdf(pdf_path)
    return build_rag_index_from_text(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        batch_size=batch_size,
    )
