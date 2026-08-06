"""
train.py
---------
One-shot pipeline build script. Run this once (or whenever the dataset
changes) to produce all artifacts the Streamlit app / agent needs:

    python train.py

Produces, under artifacts/:
    clean_resumes.parquet   - cleaned dataset
    embedder.joblib          - fitted TF-IDF+SVD embedder
    classifier.joblib        - fitted MLP category classifier
    corpus_embeddings.npy    - embeddings for every resume (for retrieval)
    corpus_meta.joblib       - metadata list aligned with the embeddings
"""
import time
import joblib
import numpy as np

from src.preprocess import load_and_clean
from src.features import Embedder
from src.classifier import CategoryClassifier

DATA_PATH = "data/resumes_dataset.jsonl"
ARTIFACTS = "artifacts"


def main():
    t0 = time.time()
    print("1/4  Loading + cleaning dataset ...")
    df = load_and_clean(DATA_PATH)
    df.to_parquet(f"{ARTIFACTS}/clean_resumes.parquet")
    print(f"     {len(df)} records, {df['Category'].nunique()} categories")

    print("2/4  Fitting embedder (TF-IDF + LSA) ...")
    embedder = Embedder(max_features=20000, n_components=256)
    embeddings = embedder.fit_transform(df["clean_text"].tolist())
    embedder.save(f"{ARTIFACTS}/embedder.joblib")
    np.save(f"{ARTIFACTS}/corpus_embeddings.npy", embeddings)
    print(f"     embeddings shape: {embeddings.shape}")

    print("3/4  Training DL category classifier (MLP) ...")
    clf = CategoryClassifier(hidden_layer_sizes=(256, 128))
    clf.fit(embeddings, df["Category"].astype(str).to_numpy())
    clf.save(f"{ARTIFACTS}/classifier.joblib")

    print("4/4  Building retrieval metadata ...")
    meta = df.apply(
        lambda r: {
            "id": r["ResumeID"],
            "category": r["Category"],
            "skills": r["extracted_skills"],
            "years": int(r["years_experience"]),
            "text": r["clean_text"][:400],
        },
        axis=1,
    ).tolist()
    joblib.dump(meta, f"{ARTIFACTS}/corpus_meta.joblib")

    print(f"\nDone in {time.time()-t0:.1f}s. Artifacts saved to {ARTIFACTS}/")


if __name__ == "__main__":
    main()
