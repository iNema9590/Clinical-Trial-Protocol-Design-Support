import re
from rank_bm25 import BM25Okapi
from typing import List, Dict


# -------------------------------------------------
# Utility: Normalization
# -------------------------------------------------

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return normalize(text).split()


# -------------------------------------------------
# BM25 Structured Retriever
# -------------------------------------------------

class BM25StructuredRetriever:
    
    def __init__(self, chunks: List, target_queries: Dict[str, str]):
        """
        chunks: list of StructuredChunk objects
        target_queries: dict mapping agent_name -> enriched query string
        """

        self.chunks = chunks
        self.target_queries = target_queries
        
        # Build corpus from structured titles
        self.corpus = [
            tokenize(chunk.full_title + " " + chunk.section_id)
            for chunk in self.chunks
        ]
        
        self.bm25 = BM25Okapi(self.corpus)

    # -------------------------------------------------
    # Retrieve top-k chunks
    # -------------------------------------------------

    def retrieve_chunks(self, agent_name: str, top_k: int = 3):
        if agent_name not in self.target_queries:
            raise ValueError(f"Agent '{agent_name}' not found in TARGET_QUERIES")

        query_text = self.target_queries[agent_name]
        tokenized_query = tokenize(query_text)

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []
        for idx in ranked_indices:
            if scores[idx] <= 0:
                continue
            results.append(self.chunks[idx])
            if len(results) >= top_k:
                break

        return results

    # -------------------------------------------------
    # Build combined context for agent
    # -------------------------------------------------

    def retrieve_context(self, agent_name: str, top_k: int = 3) -> str:
        """
        Returns a single formatted context string
        to feed directly into your structured agent.
        """

        retrieved_chunks = self.retrieve_chunks(agent_name, top_k)

        if not retrieved_chunks:
            return "No relevant sections found."

        combined_sections = []

        for chunk in retrieved_chunks:
            section_block = f"""
================================================================
Section ID: {chunk.section_id}
Section Path: {chunk.full_title}
================================================================

{chunk.content.strip()}
"""
            combined_sections.append(section_block.strip())

        return "\n\n".join(combined_sections)