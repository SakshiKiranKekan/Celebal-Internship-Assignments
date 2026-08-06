"""
agent.py
---------
The agentic layer. Rather than a single fixed function, the
HiringAssistantAgent maintains conversation state and decides, at each
turn, which of several tools to invoke:

  - evaluate_resume   : run the full ML/DL + RAG scoring pipeline
  - retrieve_evidence : pull supporting/comparison resumes from the corpus
  - explain_score      : LLM call grounded in the stored assessment
  - answer_question    : open-ended RAG-grounded Q&A about the evaluation
  - request_missing_input : if resume or JD hasn't been provided yet, ask for it
                             instead of guessing

This gives the classic agent loop: observe state -> pick a tool -> act ->
update state -> respond. It's intentionally a light, transparent
implementation (no hidden framework) so every decision is inspectable.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from .preprocess import clean_text, extract_skills
from .features import Embedder
from .classifier import CategoryClassifier
from .retriever import ResumeRetriever
from .feedback import compute_fit, FitAssessment, generate_feedback, build_grounded_prompt
from .llm import LLMClient


@dataclass
class SessionState:
    resume_text: Optional[str] = None
    resume_skills: set = field(default_factory=set)
    candidate_name: str = "Candidate"
    jd_text: Optional[str] = None
    jd_skills: set = field(default_factory=set)
    assessment: Optional[FitAssessment] = None
    chat_history: List[Dict] = field(default_factory=list)


class HiringAssistantAgent:
    def __init__(self, embedder: Embedder, classifier: CategoryClassifier,
                 retriever: ResumeRetriever, llm: LLMClient):
        self.embedder = embedder
        self.classifier = classifier
        self.retriever = retriever
        self.llm = llm
        self.state = SessionState()

    # ---------- tools ----------

    def set_resume(self, resume_text: str, candidate_name: str = "Candidate"):
        self.state.resume_text = clean_text(resume_text)
        self.state.resume_skills = set(extract_skills(self.state.resume_text))
        self.state.candidate_name = candidate_name or "Candidate"
        self.state.assessment = None  # invalidate previous evaluation

    def set_job_description(self, jd_text: str):
        self.state.jd_text = jd_text
        self.state.jd_skills = set(extract_skills(clean_text(jd_text)))
        self.state.assessment = None

    def evaluate_resume(self) -> FitAssessment:
        """Runs the full pipeline: embed -> classify (DL) -> retrieve (RAG) -> score -> explain (LLM)."""
        if not self.state.resume_text:
            raise ValueError("No resume loaded. Call set_resume() first.")

        resume_vec = self.embedder.transform([self.state.resume_text])[0]

        # DL classification: predicted job category + confidence
        top = self.classifier.top_k(resume_vec, k=1)
        predicted_category, confidence = top[0]

        # Semantic similarity vs JD (0 if no JD given yet)
        if self.state.jd_text:
            jd_clean = clean_text(self.state.jd_text)
            jd_vec = self.embedder.transform([jd_clean])[0]
            semantic_sim = float(resume_vec @ jd_vec)
        else:
            semantic_sim = 0.0

        fit = compute_fit(self.state.resume_skills, self.state.jd_skills, semantic_sim)

        # RAG evidence: similar resumes + common skills in the predicted category
        similar = self.retriever.query(resume_vec, k=5, category_filter=predicted_category)
        common_skills = self.retriever.category_skill_frequency(predicted_category, top_n=15)

        assessment = FitAssessment(
            overall_score=fit["overall_score"],
            semantic_similarity=semantic_sim,
            skill_overlap=fit["skill_overlap"],
            matched_skills=fit["matched_skills"],
            missing_skills=fit["missing_skills"],
            predicted_category=predicted_category,
            category_confidence=confidence,
            similar_strong_resumes=similar,
            category_common_skills=common_skills,
        )
        assessment.feedback_text = generate_feedback(
            assessment, self.state.candidate_name,
            self.state.jd_text or "(no job description provided — general resume review)",
            self.llm,
        )
        self.state.assessment = assessment
        return assessment

    def answer_question(self, question: str) -> str:
        """Agentic Q&A: grounds the answer in the current assessment + retrieved
        evidence rather than answering from ungrounded LLM knowledge."""
        self.state.chat_history.append({"role": "user", "content": question})

        if self.state.assessment is None:
            reply = ("I don't have an evaluation yet — please share a resume "
                      "(and ideally a job description) first so I can give you "
                      "grounded, specific feedback rather than a generic answer.")
            self.state.chat_history.append({"role": "assistant", "content": reply})
            return reply

        a = self.state.assessment
        system = (
            "You are a hiring assistant chatbot talking directly to a job "
            "candidate about their resume evaluation. Answer ONLY using the "
            "evidence below plus the prior conversation. If asked something "
            "the evidence can't answer, say so honestly rather than guessing. "
            "Be concise and conversational."
        )
        evidence = f"""Current evaluation evidence:
- Overall score: {a.overall_score}/100
- Predicted category: {a.predicted_category} ({a.category_confidence:.0%} confidence)
- Matched skills: {', '.join(a.matched_skills) or 'none'}
- Missing skills: {', '.join(a.missing_skills) or 'none'}
- Common skills among strong resumes in this category: {', '.join(s for s,_ in a.category_common_skills[:10])}

Conversation so far:
{self._format_history()}

Candidate's question: {question}"""
        answer = self.llm.generate(system, evidence, max_tokens=400)
        self.state.chat_history.append({"role": "assistant", "content": answer})
        return answer

    def _format_history(self) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.state.chat_history[-6:])

    # ---------- top-level agent loop ----------

    def handle_turn(self, user_message: str) -> str:
        """Simple planning step: decide what the agent needs before it can help."""
        if self.state.resume_text is None:
            return ("I don't have a resume yet. Please paste the resume text "
                     "(or upload it) and I'll get started.")
        if self.state.assessment is None:
            self.evaluate_resume()
            return self.state.assessment.feedback_text
        return self.answer_question(user_message)
