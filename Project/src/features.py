"""
features.py
------------
Classical ML feature layer: TF-IDF -> Truncated SVD (LSA) to get dense
semantic embeddings for resumes and job descriptions without needing
external pretrained embedding downloads. This is the vector space the
retriever (RAG) and the neural classifier both consume.

Swap-in point: if you have network access to a hosted embedding model
(OpenAI/Anthropic/HF), replace `Embedder.transform` with an API call and
everything downstream (retriever, classifier, agent) keeps working, since
they only depend on `Embedder.transform(list[str]) -> np.ndarray`.
"""
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class Embedder:
    def __init__(self, max_features=20000, n_components=256, random_state=42):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.is_fit = False

    def fit(self, texts):
        tfidf = self.vectorizer.fit_transform(texts)
        self.svd.fit(tfidf)
        self.is_fit = True
        return self

    def transform(self, texts) -> np.ndarray:
        if not self.is_fit:
            raise RuntimeError("Embedder must be fit before transform")
        tfidf = self.vectorizer.transform(texts)
        emb = self.svd.transform(tfidf)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms

    def fit_transform(self, texts) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)

    def save(self, path):
        joblib.dump({"vectorizer": self.vectorizer, "svd": self.svd}, path)

    @classmethod
    def load(cls, path):
        obj = cls()
        data = joblib.load(path)
        obj.vectorizer = data["vectorizer"]
        obj.svd = data["svd"]
        obj.is_fit = True
        return obj


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (n,d) normalized, b: (m,d) normalized -> (n,m) similarity matrix"""
    return a @ b.T
