import os
import numpy as np
import streamlit as st
from datetime import datetime

from src.features import Embedder
from src.classifier import CategoryClassifier
from src.retriever import ResumeRetriever
from src.llm import LLMClient
from src.agent import HiringAssistantAgent

ARTIFACTS = "artifacts"

# Page configuration
st.set_page_config(
    page_title="Intelligent Hiring Assistant", 
    page_icon="🧑‍💼", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem 1rem;
    }
    
    /* Card-like containers */
    .stTextArea > div > div {
        background-color: #f8f9fa;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    
    .stTextArea > div > div:focus-within {
        border-color: #4CAF50;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2);
    }
    
    /* Custom metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
        margin: 0.5rem 0;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card.green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .metric-card.orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .metric-card.blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .metric-card.purple {
        background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%);
    }
    
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.9;
        margin-bottom: 0.25rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    
    .metric-sub {
        font-size: 0.8rem;
        opacity: 0.8;
        margin-top: 0.25rem;
    }
    
    /* Chat styling */
    .chat-message {
        padding: 1rem 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
    }
    
    .chat-message.user {
        background: #f0f7ff;
        border-left: 4px solid #4CAF50;
    }
    
    .chat-message.assistant {
        background: white;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 1.1rem;
        font-weight: 600;
        background-color: #f8f9fa;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .status-badge.online {
        background: #d4edda;
        color: #155724;
    }
    
    .status-badge.offline {
        background: #f8d7da;
        color: #721c24;
    }
    
    /* Info box */
    .info-box {
        background: #e8f5e9;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Skills tags */
    .skill-tag {
        display: inline-block;
        background: #e3f2fd;
        color: #1565c0;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.15rem;
        border: 1px solid #bbdefb;
    }
    
    .skill-tag.missing {
        background: #ffebee;
        color: #c62828;
        border-color: #ffcdd2;
    }
    
    /* Divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(to right, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_agent(api_key: str = None):
    embedder = Embedder.load(f"{ARTIFACTS}/embedder.joblib")
    classifier = CategoryClassifier.load(f"{ARTIFACTS}/classifier.joblib")
    embeddings = np.load(f"{ARTIFACTS}/corpus_embeddings.npy")
    import joblib
    meta = joblib.load(f"{ARTIFACTS}/corpus_meta.joblib")
    retriever = ResumeRetriever(embeddings, meta)
    llm = LLMClient(api_key=api_key)
    return HiringAssistantAgent(embedder, classifier, retriever, llm)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">🧑‍💼</div>
        <h2 style="margin: 0; color: #333;">Hiring Assistant</h2>
        <p style="color: #666; font-size: 0.9rem;">AI-Powered Resume Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    api_key_input = st.text_input(
        "🔑 Anthropic API Key", 
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Optional: Enter your API key for enhanced LLM feedback"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🔄 Reload Pipeline", use_container_width=True):
            st.cache_resource.clear()
            st.success("Pipeline reloaded!")
    
    agent = load_agent(api_key_input or None)
    
    # Status indicator
    status = "🟢 Online" if agent.llm.enabled else "🟡 Offline (Rule-based)"
    st.markdown(f"""
    <div style="background: {'#d4edda' if agent.llm.enabled else '#fff3cd'}; 
                padding: 0.5rem 1rem; 
                border-radius: 8px; 
                margin: 1rem 0;">
        <span style="font-weight: 600;">LLM Status:</span> {status}
        <br><small style="color: #666;">{'' if agent.llm.enabled else 'Using fallback generator'}</small>
    </div>
    """, unsafe_allow_html=True)
    
    candidate_name = st.text_input("👤 Candidate Name", value="Candidate")
    
    st.markdown("---")
    
    # Quick stats section
    st.markdown("""
    <div style="font-size: 0.85rem; color: #666;">
        <p>📊 <strong>Pipeline Overview</strong></p>
        <ul style="list-style-type: none; padding-left: 0;">
            <li style="margin: 0.5rem 0;">1️⃣ Skill Extraction</li>
            <li style="margin: 0.5rem 0;">2️⃣ ML Embedding</li>
            <li style="margin: 0.5rem 0;">3️⃣ Neural Classification</li>
            <li style="margin: 0.5rem 0;">4️⃣ RAG Retrieval</li>
            <li style="margin: 0.5rem 0;">5️⃣ LLM Generation</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ---------------- Main layout ----------------
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">
        🤖 AI-Powered Hiring Assistant
    </h1>
    <p style="color: #666; font-size: 1.1rem;">
        Intelligent resume evaluation with explainable AI
    </p>
</div>
""", unsafe_allow_html=True)

# Input section with improved styling
st.markdown("### 📄 Input Documents")
st.markdown("Paste the resume and optional job description below")

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown("**📋 Resume Text**")
    resume_text = st.text_area(
        "", 
        height=280,
        placeholder="Paste or type the candidate's resume here...",
        label_visibility="collapsed",
        key="resume_input"
    )
    
with col2:
    st.markdown("**📝 Job Description**")
    jd_text = st.text_area(
        "", 
        height=280,
        placeholder="Paste the job description here (optional but recommended)...",
        label_visibility="collapsed",
        key="jd_input"
    )

# Evaluation button with better styling
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    eval_button = st.button(
        "🚀 Evaluate Resume", 
        type="primary",
        use_container_width=True
    )

if eval_button:
    if not resume_text.strip():
        st.warning("⚠️ Please paste a resume first.")
    else:
        agent.set_resume(resume_text, candidate_name)
        if jd_text.strip():
            agent.set_job_description(jd_text)
        
        with st.spinner("🔍 Analyzing resume, retrieving evidence, generating insights..."):
            assessment = agent.evaluate_resume()
        st.session_state["assessment"] = assessment
        st.session_state["messages"] = [{"role": "assistant", "content": assessment.feedback_text}]

if "assessment" in st.session_state:
    a = st.session_state["assessment"]
    
    st.markdown("---")
    
    # Results header
    st.markdown(f"""
    <div style="text-align: center; margin: 1rem 0;">
        <h2 style="color: #333;">📊 Evaluation Results</h2>
        <p style="color: #666;">Based on {len(a.similar_strong_resumes)} similar successful candidates</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics with custom styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Overall Fit</div>
            <div class="metric-value">{a.overall_score}/100</div>
            <div class="metric-sub">Candidate score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card green">
            <div class="metric-label">Semantic Similarity</div>
            <div class="metric-value">{a.semantic_similarity:.0%}</div>
            <div class="metric-sub">Role alignment</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card orange">
            <div class="metric-label">Skill Coverage</div>
            <div class="metric-value">{a.skill_overlap:.0%}</div>
            <div class="metric-sub">Skill match</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card blue">
            <div class="metric-label">Predicted Role</div>
            <div class="metric-value" style="font-size: 1.5rem;">{a.predicted_category}</div>
            <div class="metric-sub">{a.category_confidence:.0%} confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed analysis expander
    with st.expander("🔍 Detailed Analysis & Evidence", expanded=False):
        st.markdown("#### Skills Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**✅ Matched Skills**")
            if a.matched_skills:
                skills_html = "".join([f'<span class="skill-tag">{s}</span> ' for s in a.matched_skills[:15]])
                st.markdown(skills_html, unsafe_allow_html=True)
            else:
                st.caption("No matching skills found")
            
        with col2:
            st.markdown("**❌ Missing Skills**")
            if a.missing_skills:
                skills_html = "".join([f'<span class="skill-tag missing">{s}</span> ' for s in a.missing_skills[:15]])
                st.markdown(skills_html, unsafe_allow_html=True)
            else:
                st.caption("No missing skills identified")
        
        st.markdown("---")
        
        st.markdown(f"**📊 Common Skills for `{a.predicted_category}` Roles**")
        if a.category_common_skills:
            common_skills = [s for s, _ in a.category_common_skills[:12]]
            skills_html = "".join([f'<span class="skill-tag">{s}</span> ' for s in common_skills])
            st.markdown(skills_html, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("**📚 Similar Resumes (RAG Evidence)**")
        for idx, r in enumerate(a.similar_strong_resumes[:5], 1):
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 0.75rem 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 3px solid #667eea;">
                <strong>#{idx}</strong> {r['category']} · Similarity: {r['similarity']:.2f} · {r['years']} yrs experience
                <br><span style="font-size: 0.85rem; color: #666;">Skills: {', '.join(r['skills'][:6])}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat interface
    st.markdown("---")
    st.markdown("### 💬 Conversational Analysis")
    st.caption("Ask follow-up questions about the evaluation")
    
    # Chat container with custom styling
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.get("messages", []):
            role_class = "user" if msg["role"] == "user" else "assistant"
            icon = "👤" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"]):
                st.markdown(f"""
                <div class="chat-message {role_class}">
                    <strong>{icon} {msg['role'].capitalize()}</strong>
                    <br>{msg['content']}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask about the score, gaps, or how to improve..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your question..."):
                reply = agent.answer_question(prompt)
            st.markdown(reply)
        st.session_state["messages"].append({"role": "assistant", "content": reply})
else:
    # Welcome message with better styling
    st.markdown("""
    <div class="info-box">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div style="font-size: 2rem;">🚀</div>
            <div>
                <strong>Ready to get started!</strong><br>
                Paste a resume (and optionally a job description) above, then click 
                <strong>"Evaluate Resume"</strong> to begin the AI-powered analysis.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
            <div style="font-size: 2rem;">📊</div>
            <strong>ML Scoring</strong>
            <p style="font-size: 0.85rem; color: #666;">Neural network classification with 85%+ accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
            <div style="font-size: 2rem;">🔍</div>
            <strong>RAG Evidence</strong>
            <p style="font-size: 0.85rem; color: #666;">Similar candidates with explainable matches</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
            <div style="font-size: 2rem;">💬</div>
            <strong>LLM Insights</strong>
            <p style="font-size: 0.85rem; color: #666;">Conversational feedback and recommendations</p>
        </div>
        """, unsafe_allow_html=True)