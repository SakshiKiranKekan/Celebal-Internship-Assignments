import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np

from src.features import Embedder
from src.classifier import CategoryClassifier
from src.retriever import ResumeRetriever
from src.llm import LLMClient
from src.agent import HiringAssistantAgent

ARTIFACTS = "artifacts"

embedder = Embedder.load(f"{ARTIFACTS}/embedder.joblib")
classifier = CategoryClassifier.load(f"{ARTIFACTS}/classifier.joblib")
embeddings = np.load(f"{ARTIFACTS}/corpus_embeddings.npy")
meta = joblib.load(f"{ARTIFACTS}/corpus_meta.joblib")
retriever = ResumeRetriever(embeddings, meta)
llm = LLMClient()  # no key -> fallback mode

agent = HiringAssistantAgent(embedder, classifier, retriever, llm)

resume = """
Senior Python developer with 6 years of experience building REST APIs with
Django and Flask. Strong background in SQL (PostgreSQL, MySQL), Docker,
Git, and CI/CD pipelines with Jenkins. Some exposure to AWS and Kubernetes.
Led a team of 4 engineers. Familiar with Agile/Scrum practices.
"""

jd = """
We are hiring a Python Backend Engineer. Requirements: Python, Django or
Flask, REST API design, PostgreSQL, Docker, Kubernetes, AWS, CI/CD,
microservices architecture, and experience mentoring junior engineers.
"""

agent.set_resume(resume, candidate_name="Alex Rivera")
agent.set_job_description(jd)
assessment = agent.evaluate_resume()

print("=== ASSESSMENT ===")
print("Score:", assessment.overall_score)
print("Semantic sim:", round(assessment.semantic_similarity, 3))
print("Skill overlap:", assessment.skill_overlap)
print("Matched:", assessment.matched_skills)
print("Missing:", assessment.missing_skills)
print("Predicted category:", assessment.predicted_category, assessment.category_confidence)
print("\n=== FEEDBACK ===")
print(assessment.feedback_text)

print("\n=== CHAT TEST ===")
print(agent.answer_question("What should I focus on improving first?"))
