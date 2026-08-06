"""
feedback.py
------------
Combines the ML/DL scoring signals with retrieved evidence (RAG) to
produce an explainable fit assessment. The LLM is used to turn the
structured evidence into fluent, candidate-facing feedback — but every
claim it makes is grounded in numbers/evidence computed upstream, not
invented, which is what makes the feedback "explainable" rather than a
black-box score.
"""
from dataclasses import dataclass, field
from typing import List, Dict

from .llm import LLMClient


@dataclass
class FitAssessment:
    overall_score: float                 # 0-100
    semantic_similarity: float           # 0-1, resume vs JD embedding cosine sim
    skill_overlap: float                 # 0-1, matched / required skills
    matched_skills: List[str]
    missing_skills: List[str]
    predicted_category: str
    category_confidence: float
    similar_strong_resumes: List[Dict]   # RAG evidence
    category_common_skills: List[tuple]  # RAG evidence
    feedback_text: str = ""


def compute_fit(resume_skills: set, jd_skills: set, semantic_sim: float) -> Dict:
    if jd_skills:
        matched = sorted(resume_skills & jd_skills)
        missing = sorted(jd_skills - resume_skills)
        skill_overlap = len(matched) / max(1, len(jd_skills))
    else:
        matched, missing, skill_overlap = sorted(resume_skills), [], 1.0

    # Weighted blend: semantic similarity captures context/phrasing,
    # skill_overlap captures hard requirement coverage.
    overall = 100 * (0.45 * semantic_sim + 0.55 * skill_overlap)
    return {
        "overall_score": round(overall, 1),
        "skill_overlap": round(skill_overlap, 3),
        "matched_skills": matched,
        "missing_skills": missing,
    }


def build_grounded_prompt(assessment: FitAssessment, candidate_name: str, jd_text: str) -> tuple:
    system = (
        "You are an expert, encouraging technical recruiter assistant. "
        "You write concise, specific, and honest feedback to job candidates "
        "explaining how their resume matches a job description. "
        "You must ONLY use the evidence provided to you below — do not invent "
        "skills, experience, or numbers that are not given. "
        "Keep the tone constructive: acknowledge strengths first, then gaps, "
        "then concrete next steps. Use short paragraphs and bullet points. "
        "Never fabricate facts about the candidate."
    )
    user = f"""Candidate: {candidate_name}

JOB DESCRIPTION (excerpt):
{jd_text[:1200]}

COMPUTED EVIDENCE (ground your response in this, do not contradict it):
- Overall fit score: {assessment.overall_score}/100
- Semantic similarity between resume and job description: {assessment.semantic_similarity:.2f} (0-1 scale)
- Skill coverage of job requirements: {assessment.skill_overlap:.0%}
- Matched skills: {', '.join(assessment.matched_skills) or 'none detected'}
- Missing / gap skills: {', '.join(assessment.missing_skills) or 'none — full coverage'}
- Predicted resume category: {assessment.predicted_category} (model confidence {assessment.category_confidence:.0%})
- Skills most common among other strong "{assessment.predicted_category}" resumes in our corpus:
  {', '.join(s for s, _ in assessment.category_common_skills[:10])}

TASK:
Write personalized feedback for this candidate with three sections:
1. **Strengths** — what matches well and why (cite matched skills / score).
2. **Gaps** — what's missing relative to the JD, referencing the missing skills list.
3. **Next Steps** — 2-4 concrete, specific actions to close the gap, informed by
   the "common skills among strong resumes in this category" evidence.
Keep it under 220 words."""
    return system, user


def rule_based_feedback(assessment: FitAssessment, candidate_name: str) -> str:
    """Deterministic fallback used when no LLM is configured — still fully
    grounded in the same evidence, just template-composed instead of
    LLM-composed."""
    lines = [f"**Fit Assessment for {candidate_name}**", ""]
    lines.append(f"Overall score: **{assessment.overall_score}/100** "
                 f"(semantic match {assessment.semantic_similarity:.0%}, "
                 f"skill coverage {assessment.skill_overlap:.0%})")
    lines.append("")
    lines.append("**Strengths**")
    if assessment.matched_skills:
        lines.append("- Matches on: " + ", ".join(assessment.matched_skills[:12]))
    else:
        lines.append("- No direct skill keyword matches found; score relies on semantic similarity.")
    lines.append(f"- Resume most closely resembles the **{assessment.predicted_category}** "
                 f"category ({assessment.category_confidence:.0%} model confidence).")
    lines.append("")
    lines.append("**Gaps**")
    if assessment.missing_skills:
        lines.append("- Missing from the job requirements: " + ", ".join(assessment.missing_skills))
    else:
        lines.append("- No missing required skills detected.")
    lines.append("")
    lines.append("**Suggested next steps**")
    common = [s for s, _ in assessment.category_common_skills if s not in assessment.matched_skills][:4]
    if common:
        lines.append("- Consider highlighting or building experience with: " + ", ".join(common)
                     + " — common among strong candidates in this category.")
    if assessment.missing_skills:
        lines.append("- Add concrete examples/projects demonstrating: " + ", ".join(assessment.missing_skills[:4]))
    lines.append("- Quantify impact in your experience bullets (metrics, scale, outcomes).")
    return "\n".join(lines)


def generate_feedback(assessment: FitAssessment, candidate_name: str, jd_text: str,
                       llm: LLMClient) -> str:
    if llm.enabled:
        system, user = build_grounded_prompt(assessment, candidate_name, jd_text)
        return llm.generate(system, user)
    return rule_based_feedback(assessment, candidate_name)
