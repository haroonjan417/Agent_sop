import os
import streamlit as st
from tools import build_vectorstore_from_files
from agent import run_sop_agent

# Page Config & Bright Theme Styling
st.set_page_config(page_title="AI SOP Operations Agent", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F8FAFC; color: #0F172A; }
    section[data-testid="stSidebar"] { background-color: #F1F5F9; border-right: 1px solid #E2E8F0; }
    h1, h2, h3 { color: #0F172A !important; font-family: 'Inter', sans-serif; }
    .stTitle {
        background: linear-gradient(90deg, #EA580C 0%, #F97316 50%, #F59E0B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .hero-team-card {
        background-color: #FFFFFF;
        border: 1px solid #FFEDD5;
        border-left: 4px solid #F97316;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 10px;
        margin-bottom: 25px;
    }
    .hero-leader { color: #C2410C; font-size: 0.95rem; font-weight: 700; }
    .hero-leader span { color: #0F172A; font-weight: 600; }
    .hero-members { color: #475569; font-size: 0.9rem; }
    .hero-members span { color: #334155; font-weight: 500; }
    div[data-baseweb="textarea"] textarea, div[data-baseweb="input"] input {
        color: #0F172A !important;
        background-color: #FFFFFF !important;
        -webkit-text-fill-color: #0F172A !important;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #F97316 0%, #EA580C 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ AI SOP Operations Agent")
st.caption("Powered by CrewAI Single-Agent RAG, Groq API & Streamlit")

st.markdown("""
<div class="hero-team-card">
    <div class="hero-leader">👑 Team Leader: <span>Masood ur Rahman</span></div>
    <div class="hero-members">🤝 Team Members: <span>M Haroon Jan &nbsp;•&nbsp; Fatima Ijaz &nbsp;•&nbsp; Aslam Afridi &nbsp;•&nbsp; Shakeel Ahmad &nbsp;•&nbsp; Sami Ur Rahman</span></div>
</div>
""", unsafe_allow_html=True)

if "indexed_files_count" not in st.session_state:
    st.session_state["indexed_files_count"] = 0
if "indexed_file_names" not in st.session_state:
    st.session_state["indexed_file_names"] = []

groq_api_key = st.secrets.get("GROQ_API_KEY", "").strip()

with st.sidebar:
    st.header("⚙️ Configuration")
    if not groq_api_key:
        groq_api_key = st.text_input("Groq API Key", type="password")
    else:
        st.success("Groq API Key loaded securely")

    selected_model = st.selectbox(
        "Groq Model",
        ["groq/openai/gpt-oss-120b", "groq/llama-3.3-70b-versatile"]
    )
    
    st.markdown("---")
    st.header("📁 SOP Knowledge Base")
    uploaded_sops = st.file_uploader("Upload Company SOPs (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)

    if uploaded_sops:
        if st.button("Index / Reload SOP Knowledge Base"):
            with st.spinner("Indexing uploaded SOP documents..."):
                count = build_vectorstore_from_files(uploaded_sops)
                st.session_state["indexed_files_count"] = count
                st.session_state["indexed_file_names"] = [f.name for f in uploaded_sops]
                st.success(f"Successfully indexed {count} document(s)!")

    if st.session_state["indexed_files_count"] > 0:
        st.markdown("**Active SOPs:**")
        for fname in st.session_state["indexed_file_names"]:
            st.markdown(f"- `{fname}`")

st.markdown("### 🚨 Report an Operational Incident")
user_incident = st.text_area("Describe the operational issue or event:", height=120)

if st.button("Analyze & Generate Action Plan", type="primary"):
    if not groq_api_key:
        st.error("Missing Groq API key.")
    elif not user_incident.strip():
        st.warning("Please provide an incident description.")
    else:
        with st.spinner("SOP Agent evaluating incident..."):
            try:
                result = run_sop_agent(
                    user_incident=user_incident,
                    api_key=groq_api_key,
                    model_name=selected_model
                )
                st.success("Action Plan Ready for Approval")
                st.markdown("---")
                st.markdown(result.raw)
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
