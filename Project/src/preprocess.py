"""
preprocess.py
-------------
Cleans the raw resume dataset and extracts structured signal from noisy free text.

The source JSONL has messy fields (garbled phone/date strings, boilerplate resume
headers repeated across records, inconsistent whitespace). This module normalizes
everything into a clean DataFrame that the rest of the pipeline (feature
extraction, classifier, retriever, RAG) can rely on.
"""
import json
import re
from pathlib import Path

import pandas as pd

BOILERPLATE_PATTERNS = [
    r"jessica claire montgomery street san francisco ca \d+.*?resumesampleexamplecom",
    r"professional summary\b",
]

CANONICAL_SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "sql", "nosql",
    "html", "css", "react", "angular", "vue", "node.js", "django", "flask",
    "spring", "spring boot", ".net", "dotnet", "aws", "azure", "gcp", "docker",
    "kubernetes", "terraform", "jenkins", "ci/cd", "git", "linux", "agile",
    "scrum", "machine learning", "deep learning", "nlp", "tensorflow",
    "pytorch", "scikit-learn", "pandas", "numpy", "data science", "etl",
    "power bi", "tableau", "excel", "hadoop", "spark", "kafka", "airflow",
    "mongodb", "postgresql", "mysql", "oracle", "redis", "elasticsearch",
    "rest api", "graphql", "microservices", "selenium", "junit", "jira",
    "salesforce", "sap", "blockchain", "solidity", "android", "ios",
    "swift", "kotlin", "flutter", "react native", "figma", "sketch",
    "ui/ux", "cybersecurity", "penetration testing", "network security",
    "devops", "site reliability", "sre", "business analysis", "requirements gathering",
]


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    for pat in BOILERPLATE_PATTERNS:
        t = re.sub(pat, " ", t)
    t = re.sub(r"\S+@\S+", " ", t)                # emails
    t = re.sub(r"\b\d{5,}\b", " ", t)              # long digit noise (phones/dates jammed together)
    t = re.sub(r"[^a-z0-9\+\#\.\s]", " ", t)       # keep +,# for c++/c# and . for node.js
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extract_skills(text: str) -> list:
    t = f" {text.lower()} "
    found = []
    for skill in CANONICAL_SKILLS:
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, t):
            found.append(skill)
    return sorted(set(found))


def extract_years_experience(text: str) -> int:
    """Rough heuristic: count distinct 4-digit years in plausible range, use span."""
    years = [int(y) for y in re.findall(r"(19[9]\d|20[0-2]\d)", text)]
    years = [y for y in years if 1990 <= y <= 2026]
    if not years:
        return 0
    return max(0, max(years) - min(years))


def load_and_clean(jsonl_path: str) -> pd.DataFrame:
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(records)

    # Clean the free-text fields we can trust for signal
    df["clean_text"] = (df["Text"].fillna("") + " " + df["Skills"].fillna("")).apply(clean_text)
    df["clean_summary"] = df["Summary"].fillna("").apply(clean_text)
    df["extracted_skills"] = df["clean_text"].apply(extract_skills)
    df["years_experience"] = df["clean_text"].apply(extract_years_experience)
    df["skill_count"] = df["extracted_skills"].apply(len)

    # Drop rows with no usable text
    df = df[df["clean_text"].str.len() > 30].reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = load_and_clean("data/resumes_dataset.jsonl")
    print(f"Loaded {len(df)} clean records across {df['Category'].nunique()} categories")
    print(df[["ResumeID", "Category", "extracted_skills", "years_experience"]].head(5).to_string())
    Path("artifacts").mkdir(exist_ok=True)
    df.to_parquet("artifacts/clean_resumes.parquet")
    print("Saved artifacts/clean_resumes.parquet")
