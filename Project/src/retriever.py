"""
retriever.py
-------------
The 'R' in RAG. Wraps a fitted embedding space + a nearest-neighbor index
over the resume corpus so the agent can pull grounding evidence: e.g.
"here are 5 successful resumes in this category, here's what similar
strong candidates included that this one is missing."

Uses sklearn NearestNeighbors (cosine) instead of FAISS to keep the
dependency footprint tiny; swap in FAISS/pgvector/etc. for larger corpora
without changing the calling code (same `.query()` interface).
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


class ResumeRetriever:
    def __init__(self, embeddings: np.ndarray, metadata: list):
        """
        embeddings: (n, d) L2-normalized matrix aligned with `metadata`
        metadata: list of dicts, one per resume, e.g.
            {"id":..., "category":..., "skills":[...], "years":..., "text":...}
        """
        self.embeddings = embeddings
        self.metadata = metadata
        self.index = NearestNeighbors(n_neighbors=min(20, len(metadata)), metric="cosine")
        self.index.fit(embeddings)

    def query(self, query_vec: np.ndarray, k=5, category_filter: str = None):
        query_vec = query_vec.reshape(1, -1)
        n_fetch = min(len(self.metadata), max(k * 5, k))
        dist, idx = self.index.kneighbors(query_vec, n_neighbors=n_fetch)
        results = []
        for d, i in zip(dist[0], idx[0]):
            meta = self.metadata[i]
            if category_filter and meta.get("category") != category_filter:
                continue
            results.append({**meta, "similarity": 1 - d})
            if len(results) >= k:
                break
        return results

    def category_skill_frequency(self, category: str, top_n=15):
        """Aggregate the most common skills among resumes in a category —
        used as grounding evidence for 'what strong candidates in this
        role typically have'."""
        from collections import Counter
        counter = Counter()
        for m in self.metadata:
            if m.get("category") == category:
                counter.update(m.get("skills", []))
        return counter.most_common(top_n)
