import re
import numpy as np
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
    # Retrieve chunks by ranking strategy
    # -------------------------------------------------

    def _get_ranked_indices_and_scores(self, tokenized_query: List[str]):
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )
        return ranked_indices, scores

    def _select_indices_top_k(self, ranked_indices: List[int], scores, top_k: int) -> List[int]:
        selected = []
        for idx in ranked_indices:
            if scores[idx] <= 0:
                continue
            selected.append(idx)
            if len(selected) >= top_k:
                break
        return selected

    def _select_indices_iqr(self, ranked_indices: List[int], scores) -> List[int]:
        positive_scores = [float(scores[idx]) for idx in ranked_indices if scores[idx] > 0]
        if not positive_scores:
            return []

        score_array = np.array(positive_scores, dtype=float)
        q1 = np.percentile(score_array, 25)
        q3 = np.percentile(score_array, 75)
        iqr = q3 - q1

        threshold = q3 + 1.5 * iqr if iqr > 0 else q3

        selected = [idx for idx in ranked_indices if scores[idx] > 0 and scores[idx] >= threshold]

        if not selected:
            selected = [idx for idx in ranked_indices if scores[idx] > 0 and scores[idx] >= q3]

        if not selected:
            top_positive = next((idx for idx in ranked_indices if scores[idx] > 0), None)
            if top_positive is not None:
                selected = [top_positive]

        return selected

    def retrieve_chunks(self, agent_name: str, top_k: int = 3, selection_mode: str = "iqr"):
        if agent_name not in self.target_queries:
            raise ValueError(f"Agent '{agent_name}' not found in TARGET_QUERIES")

        query_text = self.target_queries[agent_name]
        tokenized_query = tokenize(query_text)

        ranked_indices, scores = self._get_ranked_indices_and_scores(tokenized_query)

        if selection_mode == "top_k":
            selected_indices = self._select_indices_top_k(ranked_indices, scores, top_k)
        elif selection_mode == "iqr":
            selected_indices = self._select_indices_iqr(ranked_indices, scores)
        else:
            raise ValueError("selection_mode must be either 'iqr' or 'top_k'")

        return [self.chunks[idx] for idx in selected_indices]

    # -------------------------------------------------
    # Build combined context for agent
    # -------------------------------------------------

    def retrieve_context(self, agent_name: str, top_k: int = 3, selection_mode: str = "iqr") -> str:
        """
        Returns a single formatted context string
        to feed directly into your structured agent.
        """

        retrieved_chunks = self.retrieve_chunks(
            agent_name=agent_name,
            top_k=top_k,
            selection_mode=selection_mode,
        )

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