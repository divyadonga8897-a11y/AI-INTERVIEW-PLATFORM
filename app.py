import streamlit as st
import os
import time
import cv2
import numpy as np
from datetime import datetime
import sqlite3
from reportlab.lib import colors
st.title("AI Placement Advisor & Interview Conductor")

# Initialize database
from database import init_db, save_user, get_user_by_email, save_resume_analysis, get_latest_resume_analysis
from database import create_interview, update_interview, add_interview_question, save_interview_answer, get_interview_transcript

# Import utilities
from utils.resume_parser import extract_text_from_pdf, parse_resume_to_json
from utils.ats_engine import analyze_resume_ats
from utils.resume_builder import enhance_bullet_point, generate_pdf_resume
from utils.audio_handler import generate_tts_audio, process_and_transcribe_mic
from utils.video_handler import VideoAnalyzer
from utils.interview_engine import generate_interview_questions, evaluate_candidate_answer, compile_final_interview_report

# Ensure assets directories exist
os.makedirs("assets/audio", exist_ok=True)
os.makedirs("assets/resumes", exist_ok=True)
os.makedirs("assets/reports", exist_ok=True)

# Page Configuration
st.set_page_config(
    page_title="AI Placement Advisor & Interview Conductor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Dark Theme & Neon Accents)
st.markdown("""
    <style>
        /* CSS Styling for AI Placement Advisor */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
            color: #f8fafc;
        }
        
        .main {
            background-color: #0b0f19;
        }
        
        /* Metric cards styling */
        .ats-metric-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 20px 0;
        }
        
        .ats-circle {
            width: 140px;
            height: 140px;
            border-radius: 50%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            border: 6px solid #10b981;
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
            background: radial-gradient(circle, #064e3b 0%, #022c22 100%);
        }
        
        .ats-circle-value {
            font-size: 38px;
            font-weight: 700;
            color: #34d399;
        }
        
        .ats-circle-label {
            font-size: 11px;
            color: #a7f3d0;
            letter-spacing: 1px;
            text-transform: uppercase;
        }
        
        /* Roadmap Timeline styling */
        .timeline {
            border-left: 3px solid #3b82f6;
            margin-left: 20px;
            padding-left: 20px;
            position: relative;
        }
        
        .timeline-card {
            background-color: #1e293b;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            border-left: 4px solid #10b981;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .timeline-title {
            font-weight: 600;
            font-size: 16px;
            color: #60a5fa;
            margin-bottom: 5px;
        }
        
        .timeline-body {
            font-size: 13px;
            color: #94a3b8;
        }
        
        /* Animated Avatar */
        .avatar-box {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 200px;
            background-color: #0f172a;
            border-radius: 12px;
            border: 1px solid #1e293b;
            position: relative;
            overflow: hidden;
        }
        
        .avatar-pulse {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            background: radial-gradient(circle, #3b82f6 0%, #1d4ed8 70%);
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.6);
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .avatar-speaking {
            animation: pulse-animation-speaking 1.2s infinite alternate;
            background: radial-gradient(circle, #10b981 0%, #047857 70%);
            box-shadow: 0 0 30px rgba(16, 185, 129, 0.8);
        }
        
        .avatar-thinking {
            animation: pulse-animation-thinking 1.5s infinite linear;
            background: radial-gradient(circle, #8b5cf6 0%, #6d28d9 70%);
            box-shadow: 0 0 30px rgba(139, 92, 246, 0.8);
        }
        
        @keyframes pulse-animation-speaking {
            0% { transform: scale(0.9); box-shadow: 0 0 10px rgba(16, 185, 129, 0.4); }
            100% { transform: scale(1.15); box-shadow: 0 0 40px rgba(16, 185, 129, 0.9); }
        }
        
        @keyframes pulse-animation-thinking {
            0% { transform: rotate(0deg) scale(1); }
            100% { transform: rotate(360deg) scale(1.05); }
        }
        
        .avatar-face {
            color: #ffffff;
            font-size: 32px;
            font-weight: bold;
        }
        
        /* Video frame wrapper */
        .video-wrapper {
            border: 2px solid #334155;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
            background-color: #000;
        }
        
        /* Headers & badges */
        .badge {
            background-color: #1e293b;
            border: 1px solid #3b82f6;
            color: #3b82f6;
            font-size: 11px;
            padding: 3px 8px;
            border-radius: 12px;
            font-weight: 500;
            display: inline-block;
            margin-right: 5px;
            margin-bottom: 5px;
        }
        
        .badge-green {
            border-color: #10b981;
            color: #10b981;
        }
        
        .transcript-container {
            max-height: 400px;
            overflow-y: auto;
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 10px;
        }
        
        .qa-block {
            border-bottom: 1px solid #1e293b;
            padding: 10px 0;
        }
        
        .qa-question {
            color: #38bdf8;
            font-size: 13px;
            font-weight: 500;
        }
        
        .qa-answer {
            color: #cbd5e1;
            font-size: 12px;
            margin-top: 4px;
            font-style: italic;
        }
        
        .qa-feedback {
            color: #94a3b8;
            font-size: 11px;
            margin-top: 4px;
        }
    </style>
""", unsafe_allow_html=True)

# DB Initialise
try:
    init_db()
except Exception as e:
    st.error(f"Database Initialization failed: {e}")

# Session State Initializations
if "api_provider" not in st.session_state:
    st.session_state.api_provider = "groq"
if "parsed_resume" not in st.session_state:
    st.session_state.parsed_resume = {}
if "ats_report" not in st.session_state:
    st.session_state.ats_report = {}
if "uploaded_resume_text" not in st.session_state:
    st.session_state.uploaded_resume_text = ""
if "resume_uploaded" not in st.session_state:
    st.session_state.resume_uploaded = False
if "interview_id" not in st.session_state:
    st.session_state.interview_id = None
if "interview_questions" not in st.session_state:
    st.session_state.interview_questions = []
if "current_question_idx" not in st.session_state:
    st.session_state.current_question_idx = 0
if "interview_transcript" not in st.session_state:
    st.session_state.interview_transcript = []
if "interview_status" not in st.session_state:
    st.session_state.interview_status = "NOT_STARTED" # NOT_STARTED, TECHNICAL_ROUND, HR_ROUND, COMPLETED, FAILED
if "interviewer_state" not in st.session_state:
    st.session_state.interviewer_state = "listening" # speaking, listening, thinking
if "active_tts_file" not in st.session_state:
    st.session_state.active_tts_file = ""
if "webcam_enabled" not in st.session_state:
    st.session_state.webcam_enabled = False
if "eye_contact_history" not in st.session_state:
    st.session_state.eye_contact_history = []
if "confidence_history" not in st.session_state:
    st.session_state.confidence_history = []
if "expressions_history" not in st.session_state:
    st.session_state.expressions_history = []
if "final_evaluation" not in st.session_state:
    st.session_state.final_evaluation = {}

# --- SIDEBAR PANELS ---
with st.sidebar:
    st.title("⚙️ AI Configuration")
    
    # Provider Selection
    st.session_state.api_provider = st.selectbox(
        "AI API Provider",
        ["groq", "openai"],
        index=0,
        help="Select the API backend. Groq runs Llama 3.3 and Whisper-large-v3 at near-instant speeds."
    )
    
    # Key Entry
    if st.session_state.api_provider == "groq":
        # Load from .env if present
        env_key = os.getenv("GROQ_API_KEY", "")
        api_key = st.text_input("Groq API Key", value=env_key, type="password")
    else:
        env_key = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input("OpenAI API Key", value=env_key, type="password")
        
    st.markdown("---")
    
    # User Profile / Email (For database storage linking)
    st.subheader("👤 Candidate Profile")
    user_email = st.text_input("Email Address", value="candidate@gmail.com", help="Used to store and sync resume history and mock sessions.")
    
    # Set Keys in Environment if provided
    if api_key:
        if st.session_state.api_provider == "groq":
            os.environ["GROQ_API_KEY"] = api_key
        else:
            os.environ["OPENAI_API_KEY"] = api_key
            
    st.markdown("---")
    st.caption("AI Placement Advisor v1.0 • Built on Gemini & Streamlit")

# Main Dashboard Routing
tabs = st.tabs(["Resume Analyzer 🔍", "Resume Builder 📝", "AI Interview Conductor 🎤"])

# ==========================================
# 1. RESUME ANALYZER TAB
# ==========================================
with tabs[0]:
    st.markdown("<h2 style='text-align: center; color: #60a5fa;'>🔍 AI Resume Analyzer & ATS Benchmarking</h2>", unsafe_allow_html=True)
    st.write("Upload your PDF resume to parse skills, check ATS match scores, discover missing skills, and build a personalized 6-month career preparation roadmap.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📤 Upload Resume")
        uploaded_file = st.file_uploader("Select Resume (PDF)", type=["pdf"], key="analyzer_file_uploader")
        
        target_role = st.selectbox(
            "Target Job Role",
            ["Python Developer", "Backend Developer", "Data Analyst", "Machine Learning Engineer", "Frontend Developer", "Full Stack Developer", "Data Scientist"],
            index=0
        )
        
        analyze_btn = st.button("🚀 Benchmarking & Report", use_container_width=True)
        
        if analyze_btn:
            if not uploaded_file:
                st.warning("Please upload a PDF resume file first!")
            elif not api_key:
                st.error("Please configure your API key in the sidebar configuration first.")
            else:
                with st.spinner("Step 1: Extracting raw resume content..."):
                    raw_text = extract_text_from_pdf(uploaded_file)
                    st.session_state.uploaded_resume_text = raw_text
                    
                if raw_text:
                    with st.spinner("Step 2: Parsing resume fields via Intelligence Engine..."):
                        parsed_profile = parse_resume_to_json(raw_text, api_key, st.session_state.api_provider)
                        st.session_state.parsed_resume = parsed_profile
                        st.session_state.resume_uploaded = True
                        
                        # Save candidate details to DB
                        parsed_profile["email"] = user_email
                        try:
                            save_user(parsed_profile)
                        except Exception as ex:
                            print(f"Db save user failed: {ex}")
                            
                    with st.spinner("Step 3: Calculating ATS Scores & Skills Matching..."):
                        ats_report = analyze_resume_ats(parsed_profile, target_role, api_key, st.session_state.api_provider)
                        st.session_state.ats_report = ats_report
                        
                        # Save Report to DB
                        ats_report["email"] = user_email
                        ats_report["file_name"] = uploaded_file.name
                        try:
                            save_resume_analysis(ats_report)
                        except Exception as ex:
                            print(f"Db save report failed: {ex}")
                            
                    st.success("ATS Analysis completed successfully!")
                else:
                    st.error("Could not extract text from this PDF. Please check the file formatting.")
                    
    with col2:
        # Render ATS Report results
        if st.session_state.ats_report:
            report = st.session_state.ats_report
            score = report.get("ats_score", 0)
            
            # Custom Color coding for score
            score_color = "#10b981" if score >= 80 else ("#f59e0b" if score >= 60 else "#ef4444")
            score_bg = "#064e3b" if score >= 80 else ("#78350f" if score >= 60 else "#7f1d1d")
            
            # Big visual gauge and headers
            col_sc, col_hd = st.columns([1, 3])
            with col_sc:
                st.markdown(f"""
                    <div class="ats-metric-container">
                        <div class="ats-circle" style="border-color: {score_color}; background: radial-gradient(circle, {score_bg} 0%, #022c22 100%); box-shadow: 0 0 20px rgba(0,0,0,0.5);">
                            <div class="ats-circle-value" style="color: {score_color};">{score}/100</div>
                            <div class="ats-circle-label">ATS Score</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            with col_hd:
                st.markdown(f"<h3 style='margin-top: 15px;'>Benchmark Analysis for: <span style='color: #60a5fa;'>{target_role}</span></h3>", unsafe_allow_html=True)
                st.write("Our resume engine has cross-referenced your profile structure, keywords, and skill matches against the target requirements.")
            
            st.markdown("---")
            
            # Score Breakdown
            st.subheader("📊 Category Scores")
            breakdown = report.get("breakdown", {})
            b_cols = st.columns(5)
            categories = [
                ("Resume Format", "format_score"),
                ("Skills Match", "skills_score"),
                ("Projects", "projects_score"),
                ("Experience", "experience_score"),
                ("Keywords", "keywords_score")
            ]
            for idx, (label, key) in enumerate(categories):
                with b_cols[idx]:
                    val = breakdown.get(key, 70)
                    st.metric(label, f"{val}%")
                    st.progress(val / 100)
            
            st.markdown("---")
            
            # Current vs Missing Skills
            col_sk1, col_sk2 = st.columns(2)
            with col_sk1:
                st.markdown("<h4>🎯 Detected Key Skills</h4>", unsafe_allow_html=True)
                curr_sk = report.get("current_skills", [])
                if curr_sk:
                    for sk in curr_sk:
                        st.markdown(f"<span class='badge badge-green'>{sk}</span>", unsafe_allow_html=True)
                else:
                    st.caption("No skills detected from the resume.")
            with col_sk2:
                st.markdown("<h4>⚠️ Missing Skill Gaps</h4>", unsafe_allow_html=True)
                miss_sk = report.get("missing_skills", [])
                if miss_sk:
                    for sk in miss_sk:
                        st.markdown(f"<span class='badge' style='border-color: #ef4444; color: #ef4444;'>{sk}</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color: #10b981;'>✓ No major skills missing!</span>", unsafe_allow_html=True)
                    
            st.markdown("---")
            
            # Weakness Check
            st.subheader("💡 Weakness Detection")
            weaknesses = report.get("weaknesses", [])
            if weaknesses:
                for w in weaknesses:
                    st.markdown(f"<span style='color: #f59e0b;'>⚠️</span> {w}")
            else:
                st.markdown("<span style='color: #10b981;'>✓ No layout or content weaknesses detected. Great job!</span>", unsafe_allow_html=True)
                
            st.markdown("---")
            
            # Roadmap and courses
            col_rd, col_cs = st.columns([3, 2])
            with col_rd:
                st.subheader("📅 6-Month Career Preparation Roadmap")
                roadmap = report.get("roadmap", {})
                
                st.markdown("<div class='timeline'>", unsafe_allow_html=True)
                for month_num in range(1, 7):
                    m_key = f"month_{month_num}"
                    m_val = roadmap.get(m_key, f"Focus on development step {month_num}")
                    st.markdown(f"""
                        <div class="timeline-card">
                            <div class="timeline-title">Month {month_num}</div>
                            <div class="timeline-body">{m_val}</div>
                        </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_cs:
                st.subheader("📚 Recommended Courses")
                courses = report.get("recommended_courses", [])
                if courses:
                    for course in courses:
                        st.markdown(f"📖 **{course}**")
                        st.caption("Suggested focus area to bridge skill gap.")
                        st.write("")
                else:
                    st.caption("No recommended courses required.")
                
                st.subheader("💼 Suitable Job Roles")
                roles = report.get("suitable_roles", [])
                if roles:
                    for role in roles:
                        st.markdown(f"⚡ **{role}**")
                else:
                    st.caption("Target role is suitable.")
                    
        else:
            st.info("Upload your resume and click the 'Benchmarking & Report' button to run calculations.")


# ==========================================
# 2. RESUME GENERATOR TAB
# ==========================================
with tabs[1]:
    st.markdown("<h2 style='text-align: center; color: #60a5fa;'>📝 AI-Enhanced Resume Generator</h2>", unsafe_allow_html=True)
    st.write("Complete the form below, refine your bullet points using AI, select from 4 professional templates, and download a custom formatted PDF.")
    
    # Load profile data from parsed resume if available
    def_profile = st.session_state.parsed_resume or {}
    
    # Interactive multi-step tabs for forms
    form_tabs = st.tabs(["1. Personal & Objectives", "2. Education & Skills", "3. Internships & Projects", "4. Certifications & Extras"])
    
    # Form Values Initializations
    with form_tabs[0]:
        st.subheader("Personal Details")
        f_name = st.text_input("Full Name", value=def_profile.get("full_name", ""))
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            f_email = st.text_input("Email Address", value=def_profile.get("email", user_email))
            f_phone = st.text_input("Phone Number", value=def_profile.get("phone", ""))
        with col_c2:
            f_linkedin = st.text_input("LinkedIn Profile URL", value=def_profile.get("linkedin", ""))
            f_github = st.text_input("GitHub Profile URL", value=def_profile.get("github", ""))
        f_address = st.text_input("Address / Location", value=def_profile.get("address", ""))
        
        st.subheader("Career Objective")
        f_objective = st.text_area(
            "Objective statement (will be polished)", 
            value=def_profile.get("objective", ""), 
            help="Describe your career goals and what value you bring to potential employers."
        )
        
    with form_tabs[1]:
        st.subheader("Education")
        def_edu = def_profile.get("education", [])
        edu_count = st.number_input("Number of Education Degrees", min_value=1, max_value=5, value=max(1, len(def_edu)))
        
        edu_list = []
        for i in range(int(edu_count)):
            st.markdown(f"**Degree #{i+1}**")
            d_val = def_edu[i] if i < len(def_edu) else {}
            col_e1, col_e2, col_e3 = st.columns([2, 2, 1])
            with col_e1:
                e_inst = st.text_input(f"Institution / College", value=d_val.get("institution", ""), key=f"edu_inst_{i}")
            with col_e2:
                e_deg = st.text_input(f"Degree / Major", value=d_val.get("degree", ""), key=f"edu_deg_{i}")
            with col_e3:
                e_cgpa = st.text_input(f"CGPA / Percentage", value=d_val.get("cgpa", ""), key=f"edu_cgpa_{i}")
            edu_list.append({"institution": e_inst, "degree": e_deg, "cgpa": e_cgpa})
            
        st.markdown("---")
        st.subheader("Technical Skills")
        def_skills = def_profile.get("skills", [])
        skills_input = st.text_area(
            "Skills (Comma separated list)",
            value=", ".join(def_skills) if isinstance(def_skills, list) else str(def_skills),
            help="E.g., Python, Docker, SQL, Flask, Machine Learning"
        )
        
    with form_tabs[2]:
        st.subheader("Internships / Work Experience")
        def_intern = def_profile.get("internships", []) + def_profile.get("experience", [])
        intern_count = st.number_input("Number of Work Experiences", min_value=0, max_value=5, value=len(def_intern))
        
        intern_list = []
        for i in range(int(intern_count)):
            st.markdown(f"**Experience #{i+1}**")
            int_val = def_intern[i] if i < len(def_intern) else {}
            
            col_i1, col_i2, col_i3 = st.columns([2, 2, 1])
            with col_i1:
                i_comp = st.text_input("Company / Organization", value=int_val.get("company", int_val.get("organization", "")), key=f"int_comp_{i}")
            with col_i2:
                i_role = st.text_input("Role / Title", value=int_val.get("role", int_val.get("title", "")), key=f"int_role_{i}")
            with col_i3:
                i_dur = st.text_input("Duration (e.g. 3 mos)", value=int_val.get("duration", int_val.get("date", "")), key=f"int_dur_{i}")
                
            i_desc = st.text_area("Description / Duties", value=int_val.get("description", ""), key=f"int_desc_{i}")
            
            # AI Enhancement trigger for bullet points
            if st.button(f"✨ AI Enhance Experience #{i+1} bullet points", key=f"ai_enh_int_{i}"):
                if not api_key:
                    st.error("Please add your API key to proceed.")
                elif not i_desc.strip():
                    st.warning("Please enter a basic description first!")
                else:
                    with st.spinner("AI Enhancing description..."):
                        enhanced_desc = enhance_bullet_point(i_desc, api_key, st.session_state.api_provider)
                        st.info(f"Enhanced Suggestion: '{enhanced_desc}'")
                        # We save it back so user can copy/paste, or dynamically update the text field
                        st.caption("Copy the above text to replace your original description.")
            
            intern_list.append({"company": i_comp, "role": i_role, "duration": i_dur, "description": i_desc})
            st.markdown("---")
            
        st.subheader("Academic/Personal Projects")
        def_proj = def_profile.get("projects", [])
        proj_count = st.number_input("Number of Projects", min_value=0, max_value=5, value=len(def_proj))
        
        proj_list = []
        for i in range(int(proj_count)):
            st.markdown(f"**Project #{i+1}**")
            p_val = def_proj[i] if i < len(def_proj) else {}
            
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                p_title = st.text_input("Project Title", value=p_val.get("title", ""), key=f"proj_title_{i}")
            with col_p2:
                p_techs = st.text_input("Technologies (comma separated)", value=", ".join(p_val.get("technologies", [])) if isinstance(p_val.get("technologies"), list) else str(p_val.get("technologies", "")), key=f"proj_techs_{i}")
                
            p_desc = st.text_area("Description / Implementation details", value=p_val.get("description", ""), key=f"proj_desc_{i}")
            
            if st.button(f"✨ AI Enhance Project #{i+1} description", key=f"ai_enh_proj_{i}"):
                if not api_key:
                    st.error("Please configure API key first.")
                elif not p_desc.strip():
                    st.warning("Please write a project description to enhance.")
                else:
                    with st.spinner("Optimizing bullet point..."):
                        enhanced_desc = enhance_bullet_point(p_desc, api_key, st.session_state.api_provider)
                        st.info(f"Enhanced Suggestion: '{enhanced_desc}'")
                        st.caption("Copy the above text to replace your original description.")
                        
            # Parse tech strings to list
            t_list = [t.strip() for t in p_techs.split(",") if t.strip()]
            proj_list.append({"title": p_title, "technologies": t_list, "description": p_desc})
            st.markdown("---")
            
    with form_tabs[3]:
        st.subheader("Certifications & Achievements")
        def_cert = def_profile.get("certifications", [])
        def_achive = def_profile.get("achievements", [])
        
        certs_input = st.text_area(
            "Certifications (one per line)", 
            value="\n".join(def_cert) if isinstance(def_cert, list) else str(def_cert),
            help="E.g., AWS Developer Associate\nGoogle Data Analytics Professional Certificate"
        )
        
        achievs_input = st.text_area(
            "Achievements (one per line)", 
            value="\n".join(def_achive) if isinstance(def_achive, list) else str(def_achive),
            help="E.g., Secured 1st place in Inter-College Hackathon 2026\nAcademic Excellence Award recipient"
        )
        
        st.subheader("Languages")
        def_lang = def_profile.get("languages", [])
        langs_input = st.text_input(
            "Languages Spoken", 
            value=", ".join(def_lang) if isinstance(def_lang, list) else str(def_lang),
            help="E.g. English, Spanish, Hindi"
        )
        
        st.subheader("Visual Layout Options")
        col_ph1, col_ph2 = st.columns(2)
        with col_ph1:
            include_photo = st.checkbox("Include Profile Block Photo Frame", value=False)
        with col_ph2:
            template_selection = st.selectbox(
                "Select Styling Template",
                ["Modern", "Professional", "ATS Friendly", "Creative"],
                index=0,
                help="Modern layout contains sidebar. ATS Friendly is optimized for parse engines. Professional is highly traditional."
            )
            
    # PDF Compilation Button
    st.markdown("---")
    gen_pdf_btn = st.button("📄 Generate & Compile PDF Resume", use_container_width=True)
    
    if gen_pdf_btn:
        # Construct compiled profile object
        skills_arr = [s.strip() for s in skills_input.split(",") if s.strip()]
        certs_arr = [c.strip() for c in certs_input.split("\n") if c.strip()]
        achievs_arr = [a.strip() for a in achievs_input.split("\n") if a.strip()]
        langs_arr = [l.strip() for l in langs_input.split(",") if l.strip()]
        
        compiled_profile = {
            "full_name": f_name,
            "email": f_email,
            "phone": f_phone,
            "linkedin": f_linkedin,
            "github": f_github,
            "address": f_address,
            "objective": f_objective,
            "education": edu_list,
            "skills": skills_arr,
            "projects": proj_list,
            "internships": intern_list,
            "certifications": certs_arr,
            "achievements": achievs_arr,
            "languages": langs_arr,
            "has_photo": include_photo
        }
        
        # Save compiled profile back to session state to easily reload
        st.session_state.parsed_resume = compiled_profile
        st.session_state.resume_uploaded = True
        
        # Save to SQLite DB
        compiled_profile["email"] = user_email
        try:
            save_user(compiled_profile)
        except Exception as ex:
            print(f"Db save user builder failed: {ex}")
            
        with st.spinner("Compiling ReportLab flowables..."):
            filename = f"resume_{user_email.replace('@','_').replace('.','_')}.pdf"
            filepath = os.path.join("assets/resumes", filename)
            
            try:
                generate_pdf_resume(compiled_profile, template_selection, filepath)
                st.success(f"Resume generated successfully in '{template_selection}' template!")
                
                # Download Options
                with open(filepath, "rb") as pdf_file:
                    pdf_data = pdf_file.read()
                    
                st.download_button(
                    label="📥 One-Click Download PDF Resume",
                    data=pdf_data,
                    file_name=f"{f_name.replace(' ', '_')}_Resume.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
                # Mock DOCX option (offers raw text outline file)
                txt_data = f"""RESUME OUTLINE - {f_name.upper()}\n\nContact: {f_email} | {f_phone}\nLinkedIn: {f_linkedin}\nGitHub: {f_github}\nAddress: {f_address}\n\nOBJECTIVE:\n{f_objective}\n\nEDUCATION:\n"""
                for e in edu_list:
                    txt_data += f"- {e['degree']} from {e['institution']} (CGPA: {e['cgpa']})\n"
                txt_data += f"\nSKILLS: {skills_input}\n\nEXPERIENCE:\n"
                for i in intern_list:
                    txt_data += f"- {i['role']} at {i['company']} ({i['duration']})\n  {i['description']}\n"
                txt_data += f"\nPROJECTS:\n"
                for p in proj_list:
                    txt_data += f"- {p['title']} using {', '.join(p['technologies'])}\n  {p['description']}\n"
                    
                st.download_button(
                    label="📥 Download Resume Outline (TXT/DOCX)",
                    data=txt_data.encode("utf-8"),
                    file_name=f"{f_name.replace(' ', '_')}_Resume_Outline.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True
                )
            except Exception as ex:
                st.error(f"Failed to generate ReportLab PDF: {ex}")


# ==========================================
# 3. AI INTERVIEW CONDUCTOR TAB (MAIN MODULE)
# ==========================================
with tabs[2]:
    st.markdown("<h2 style='text-align: center; color: #60a5fa;'>🎤 AI Interview Conductor</h2>", unsafe_allow_html=True)
    
    # Check if a resume is parsed
    if not st.session_state.resume_uploaded:
        st.warning("⚠️ MANDATORY RESUME REQUIRED: You must upload your resume in the 'Resume Analyzer' tab or complete details in 'Resume Builder' tab before commencing the interview.")
        st.stop()
        
    # --- TRIPLE-PANEL INTERFACE ---
    left_panel, center_panel, right_panel = st.columns([1, 2, 1.5])
    
    # ------------------ LEFT PANEL ------------------
    with left_panel:
        st.subheader("📄 Resume Intelligence")
        st.write(f"**Candidate:** {st.session_state.parsed_resume.get('full_name', 'Student')}")
        
        # Scan Info Summary
        skills_found = st.session_state.parsed_resume.get("skills", [])
        projects_found = st.session_state.parsed_resume.get("projects", [])
        
        st.markdown(f"**Skills Scanned:** {len(skills_found)}")
        st.caption(", ".join(skills_found[:8]))
        
        st.markdown(f"**Projects Scanned:** {len(projects_found)}")
        st.caption(", ".join([p.get("title", "") for p in projects_found[:3]]))
        
        st.markdown("---")
        
        # Interview Control panel
        st.subheader("🛠️ Session Controls")
        job_selection = st.selectbox(
            "Select Interview Job Role",
            ["Python Developer", "Data Analyst", "AI Engineer", "Backend Developer", "Frontend Developer"],
            key="interview_role_dropdown"
        )
        
        # Start Interview Button
        if st.session_state.interview_status == "NOT_STARTED":
            start_int = st.button("🎬 Launch Live Simulation", use_container_width=True, type="primary")
            if start_int:
                if not api_key:
                    st.error("Configure API key first!")
                else:
                    with st.spinner("Scanning resume & generating dynamic interview bank..."):
                        # Save session initialization
                        int_id = create_interview(user_email)
                        st.session_state.interview_id = int_id
                        
                        # Generate dynamic question set based on scanned profile
                        questions = generate_interview_questions(
                            st.session_state.parsed_resume,
                            job_selection,
                            api_key,
                            st.session_state.api_provider
                        )
                        st.session_state.interview_questions = questions
                        st.session_state.current_question_idx = 0
                        st.session_state.interview_transcript = []
                        st.session_state.interview_status = "TECHNICAL_ROUND"
                        st.session_state.interviewer_state = "speaking"
                        
                        # Save questions list in DB
                        for idx, q in enumerate(questions):
                            try:
                                add_interview_question(int_id, q["question"], q["round_type"], q["topic"], idx)
                            except Exception as ex:
                                print(f"Db add question failed: {ex}")
                                
                        # TTS synthesize first question
                        first_q = questions[0]["question"]
                        tts_file = generate_tts_audio(first_q)
                        st.session_state.active_tts_file = tts_file
                        
                        st.rerun()
                        
        else:
            stop_int = st.button("🛑 Terminate Session", use_container_width=True)
            if stop_int:
                st.session_state.interview_status = "NOT_STARTED"
                st.session_state.interview_questions = []
                st.session_state.interview_transcript = []
                st.session_state.current_question_idx = 0
                st.session_state.webcam_enabled = False
                st.session_state.interviewer_state = "listening"
                st.rerun()
                
        # Displays round tracker
        st.markdown("---")
        st.subheader("📈 Round Tracker")
        if st.session_state.interview_status != "NOT_STARTED":
            status = st.session_state.interview_status
            if status == "TECHNICAL_ROUND":
                st.markdown("<span style='color: #38bdf8; font-weight: bold;'>⚡ ROUND 1: Technical Interview</span>", unsafe_allow_html=True)
                st.markdown("<span style='color: #94a3b8;'>HR Round: Unlocked at Score >= 70</span>", unsafe_allow_html=True)
            elif status == "HR_ROUND":
                st.markdown("<span style='color: #10b981; font-weight: bold;'>✓ ROUND 1: Technical (COMPLETED)</span>", unsafe_allow_html=True)
                st.markdown("<span style='color: #c084fc; font-weight: bold;'>⚡ ROUND 2: HR Interview (ACTIVE)</span>", unsafe_allow_html=True)
            elif status == "COMPLETED":
                st.markdown("<span style='color: #10b981; font-weight: bold;'>✓ ROUND 1: Technical (COMPLETED)</span>", unsafe_allow_html=True)
                st.markdown("<span style='color: #10b981; font-weight: bold;'>✓ ROUND 2: HR Round (COMPLETED)</span>", unsafe_allow_html=True)
                st.success("🎉 Final AI Evaluation Unlocked!")
            elif status == "FAILED":
                st.markdown("<span style='color: #ef4444; font-weight: bold;'>❌ TECHNICAL ROUND FAILED</span>", unsafe_allow_html=True)
                st.caption("Technical average score was below 70. HR Round locked.")
                
            st.markdown(f"**Question Progress:** {st.session_state.current_question_idx + 1} / {len(st.session_state.interview_questions)}")
        else:
            st.caption("Launch the simulation to begin tracking rounds.")

    # ------------------ CENTER PANEL ------------------
    with center_panel:
        st.subheader("🎭 AI Simulation Center")
        
        # 1. Avatar Visual (Displays pulse animations depending on state)
        avatar_class = "avatar-pulse"
        face_emoji = "🤖"
        state_label = "Listening"
        
        if st.session_state.interview_status != "NOT_STARTED":
            if st.session_state.interviewer_state == "speaking":
                avatar_class = "avatar-pulse avatar-speaking"
                face_emoji = "🔊"
                state_label = "Interviewer Speaking"
            elif st.session_state.interviewer_state == "thinking":
                avatar_class = "avatar-pulse avatar-thinking"
                face_emoji = "🧠"
                state_label = "Evaluating Answer..."
            else:
                avatar_class = "avatar-pulse"
                face_emoji = "🎤"
                state_label = "Listening (Speak Now)"
                
        st.markdown(f"""
            <div class="avatar-box">
                <div style="text-align: center;">
                    <div class="{avatar_class}">
                        <div class="avatar-face">{face_emoji}</div>
                    </div>
                    <div style="margin-top: 10px; font-weight: 500; font-size: 13px; color: #94a3b8;">{state_label}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.write("")
        
        # 2. Camera HUD Feed (Accesses cv2 capture and draws overlays)
        st.session_state.webcam_enabled = st.checkbox("Toggle Webcam Feed (Live Analysis HUD)", value=st.session_state.webcam_enabled)
        
        camera_placeholder = st.empty()
        
        # OpenCV Loop processing
        if st.session_state.webcam_enabled and st.session_state.interview_status != "NOT_STARTED" and st.session_state.interviewer_state == "listening":
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                analyzer = VideoAnalyzer()
                # Run for a few frames to populate camera window or render continuously
                # To prevent blocking streamlit indefinitely, we run a short 5-frame loop to render
                # and save metrics in accumulator.
                for _ in range(5):
                    ret, frame = cap.read()
                    if ret:
                        # Process frame
                        frame_hud, met = analyzer.analyze_frame(frame)
                        
                        # Accumulate visual statistics in lists
                        st.session_state.eye_contact_history.append(met['eye_contact'])
                        st.session_state.confidence_history.append(met['confidence_score'])
                        st.session_state.expressions_history.append(met['expression'])
                        
                        # Display
                        camera_placeholder.image(frame_hud, channels="BGR", use_container_width=True)
                        time.sleep(0.05)
                cap.release()
            else:
                camera_placeholder.error("Webcam not accessible or blocked by another process. Falled back to premium avatar mode.")
        elif st.session_state.webcam_enabled:
            # Display camera static scanning frame
            dummy_img = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(dummy_img, "Webcam Standby (Awaiting Response Round)", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
            camera_placeholder.image(dummy_img, channels="BGR", use_container_width=True)
            
        st.markdown("---")
        
        # 3. Live Question Text & TTS Audio Trigger
        if st.session_state.interview_status != "NOT_STARTED":
            curr_idx = st.session_state.current_question_idx
            q_list = st.session_state.interview_questions
            
            if curr_idx < len(q_list):
                curr_q = q_list[curr_idx]["question"]
                curr_topic = q_list[curr_idx]["topic"]
                curr_round = q_list[curr_idx]["round_type"]
                
                # Question Display
                st.markdown(f"**Round:** <span class='badge'>{(curr_round)}</span> **Topic:** <span class='badge badge-green'>{curr_topic}</span>", unsafe_allow_html=True)
                st.info(f"❓ **Interviewer Question:**\n\n{curr_q}")
                
                # Voice output playback
                if st.session_state.active_tts_file and os.path.exists(st.session_state.active_tts_file):
                    st.audio(st.session_state.active_tts_file, format="audio/mp3", autoplay=True)
                    # Reset TTS play trigger
                    st.session_state.active_tts_file = ""
                    
                # Voice Input Capture (Microphone)
                st.write("🎙️ **Verbal Response Input:**")
                recorded_audio = st.audio_input("Record your answer", key=f"mic_recorder_{curr_idx}")
                
                submit_ans = st.button("📤 Submit Answer", use_container_width=True, type="primary")
                
                if submit_ans:
                    if not recorded_audio:
                        st.warning("Please record your answer first by pressing the microphone icon.")
                    else:
                        st.session_state.interviewer_state = "thinking"
                        st.rerun()
                        
                # Processing submitted answer
                if st.session_state.interviewer_state == "thinking" and recorded_audio:
                    # Transcribe using Whisper
                    with st.spinner("Whisper transcribing speech response..."):
                        transcript = process_and_transcribe_mic(recorded_audio.read(), api_key, st.session_state.api_provider)
                        
                    if transcript:
                        # Evaluate Answer
                        with st.spinner("Evaluating answer accuracy & communication structure..."):
                            evaluation = evaluate_candidate_answer(
                                curr_q, 
                                transcript, 
                                curr_round, 
                                api_key, 
                                st.session_state.api_provider
                            )
                            
                        # Add video analysis metrics if recorded
                        avg_eye = int(np.mean(st.session_state.eye_contact_history)) if st.session_state.eye_contact_history else 95
                        avg_conf_v = int(np.mean(st.session_state.confidence_history)) if st.session_state.confidence_history else 90
                        
                        # Find most frequent expression
                        expr = "Neutral"
                        if st.session_state.expressions_history:
                            from collections import Counter
                            c = Counter(st.session_state.expressions_history)
                            expr = c.most_common(1)[0][0]
                            
                        # Set custom evaluation details
                        evaluation["user_answer"] = transcript
                        evaluation["speaking_speed"] = len(transcript.split()) * 2 # simulated words per min
                        evaluation["eye_contact_score"] = avg_eye
                        evaluation["expression_summary"] = expr
                        evaluation["question"] = curr_q
                        evaluation["round_type"] = curr_round
                        evaluation["topic"] = curr_topic
                        evaluation["order_index"] = curr_idx
                        
                        # Blend verbal confidence with visual confidence
                        evaluation["evaluation_confidence"] = int((evaluation.get("evaluation_confidence", 70.0) + avg_conf_v) / 2)
                        
                        # Save answer results to DB
                        # Find question DB ID
                        try:
                            conn = sqlite3.connect("placement_advisor.db")
                            conn.row_factory = sqlite3.Row
                            cur = conn.cursor()
                            cur.execute("""
                                SELECT id FROM interview_questions 
                                WHERE interview_id = ? AND order_index = ?
                            """, (st.session_state.interview_id, curr_idx))
                            q_row = cur.fetchone()
                            if q_row:
                                q_id = q_row["id"]
                                evaluation["question_id"] = q_id
                                save_interview_answer(q_id, evaluation)
                            conn.close()
                        except Exception as ex:
                            print(f"Db save answer failed: {ex}")
                            
                        # Update local transcript in session state
                        st.session_state.interview_transcript.append(evaluation)
                        
                        # Reset trackers
                        st.session_state.eye_contact_history = []
                        st.session_state.confidence_history = []
                        st.session_state.expressions_history = []
                        
                        # Move index or evaluate eligibility rules
                        next_idx = curr_idx + 1
                        q_tot = len(q_list)
                        
                        # Check eligibility logic (Score >= 70 unlocks HR Round)
                        # We evaluate technical round after 3 questions (index 2 is last tech question)
                        if curr_round == "TECHNICAL" and (next_idx >= 3 or next_idx == q_tot or q_list[next_idx]["round_type"] == "HR"):
                            # Aggregate Technical Average
                            tech_scores = [item["evaluation_accuracy"] for item in st.session_state.interview_transcript if item["round_type"] == "TECHNICAL"]
                            avg_tech = sum(tech_scores) / len(tech_scores) if tech_scores else 0
                            
                            if avg_tech < 70:
                                st.session_state.interview_status = "FAILED"
                                st.session_state.interviewer_state = "listening"
                                # Compile final report
                                final_report = compile_final_interview_report(st.session_state.interview_transcript)
                                st.session_state.final_evaluation = final_report
                                
                                # Update interview in DB
                                update_interview(st.session_state.interview_id, {
                                    "status": "FAILED",
                                    "overall_score": final_report["overall_score"],
                                    "technical_score": final_report["technical_score"],
                                    "hr_score": final_report["hr_score"],
                                    "communication_score": final_report["communication_score"],
                                    "confidence_score": final_report["confidence_score"],
                                    "strengths": final_report["strengths"],
                                    "weaknesses": final_report["weaknesses"],
                                    "suggestions": final_report["suggestions"]
                                })
                                st.rerun()
                            else:
                                st.session_state.interview_status = "HR_ROUND"
                                
                        if next_idx < q_tot and st.session_state.interview_status != "FAILED":
                            # Proceed to next question
                            st.session_state.current_question_idx = next_idx
                            st.session_state.interviewer_state = "speaking"
                            
                            # TTS audio pre-gen
                            next_q = q_list[next_idx]["question"]
                            tts_file = generate_tts_audio(next_q)
                            st.session_state.active_tts_file = tts_file
                            st.rerun()
                        else:
                            # Interview ends (Completed both rounds)
                            if st.session_state.interview_status != "FAILED":
                                st.session_state.interview_status = "COMPLETED"
                                final_report = compile_final_interview_report(st.session_state.interview_transcript)
                                st.session_state.final_evaluation = final_report
                                
                                # Update DB
                                update_interview(st.session_state.interview_id, {
                                    "status": "COMPLETED",
                                    "overall_score": final_report["overall_score"],
                                    "technical_score": final_report["technical_score"],
                                    "hr_score": final_report["hr_score"],
                                    "communication_score": final_report["communication_score"],
                                    "confidence_score": final_report["confidence_score"],
                                    "strengths": final_report["strengths"],
                                    "weaknesses": final_report["weaknesses"],
                                    "suggestions": final_report["suggestions"]
                                })
                            st.session_state.interviewer_state = "listening"
                            st.rerun()
                    else:
                        st.error("Whisper failed to transcribe answer. Please repeat your response.")
                        st.session_state.interviewer_state = "listening"
                        st.rerun()
                        
            else:
                st.success("🏁 Simulation completed! Review your final placement advice report in the right panel.")
        else:
            st.info("Configure your API credentials in the sidebar and launch the mock simulation to begin.")

    # ------------------ RIGHT PANEL ------------------
    with right_panel:
        st.subheader("📊 Session Diagnostics")
        
        # Active Transcript Log
        st.markdown("<h4>Transcript & Feedback</h4>", unsafe_allow_html=True)
        if st.session_state.interview_transcript:
            st.markdown("<div class='transcript-container'>", unsafe_allow_html=True)
            for idx, qa in enumerate(st.session_state.interview_transcript):
                st.markdown(f"""
                    <div class="qa-block">
                        <div class="qa-question">Q{idx+1}: {qa['question']}</div>
                        <div class="qa-answer">A: "{qa['user_answer']}"</div>
                        <div class="qa-feedback">🎯 Accuracy: {qa['evaluation_accuracy']}% | Feedback: {qa['feedback']}</div>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.caption("Active transcription & answer scoring will display here in real-time.")
            
        st.markdown("---")
        
        # Real-time Scores
        st.markdown("<h4>Evaluation Scores</h4>", unsafe_allow_html=True)
        if st.session_state.interview_transcript:
            last_qa = st.session_state.interview_transcript[-1]
            st.metric("Last Question Accuracy", f"{last_qa['evaluation_accuracy']}%")
            
            col_sc1, col_sc2 = st.columns(2)
            with col_sc1:
                st.write("**Speaking Confidence:**")
                st.progress(last_qa['evaluation_confidence'] / 100)
                st.caption(f"{last_qa['evaluation_confidence']}% ({last_qa.get('expression_summary', 'Neutral')})")
            with col_sc2:
                st.write("**Eye Contact Score:**")
                st.progress(last_qa['eye_contact_score'] / 100)
                st.caption(f"{last_qa['eye_contact_score']}%")
                
            # Verbal Fillers detected
            fillers_det = last_qa.get("fillers_detected", [])
            if fillers_det:
                st.markdown("**Detected Filler Warnings:**")
                for f_info in fillers_det:
                    st.markdown(f"<span style='color: #f59e0b;'>⚠️ Spoke '{f_info['filler']}' ({f_info['count']} times)</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color: #10b981;'>✓ Minimal verbal fillers used</span>", unsafe_allow_html=True)
        else:
            st.caption("Detailed scores, visual indicators, and vocal analysis metrics will display here.")
            
        st.markdown("---")
        
        # Final Report & Strengths/Weaknesses (Unlocked when COMPLETED or FAILED)
        if st.session_state.interview_status in ["COMPLETED", "FAILED"] and st.session_state.final_evaluation:
            final_report = st.session_state.final_evaluation
            st.markdown("<h3>📄 Final Placement Evaluation</h3>", unsafe_allow_html=True)
            
            ov_score = final_report.get("overall_score", 0)
            st.metric("OVERALL SCORE", f"{ov_score}/100")
            
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.write(f"Technical Round: **{final_report.get('technical_score')}%**")
                st.write(f"HR Round: **{final_report.get('hr_score')}%**")
            with col_r2:
                st.write(f"Communication: **{final_report.get('communication_score')}%**")
                st.write(f"Confidence Rating: **{final_report.get('confidence_score')}%**")
                
            st.markdown("---")
            st.markdown("<h4>✓ Strengths</h4>", unsafe_allow_html=True)
            for strength in final_report.get("strengths", []):
                st.write(strength)
                
            st.markdown("<h4>✗ Weaknesses & Gaps</h4>", unsafe_allow_html=True)
            for weakness in final_report.get("weaknesses", []):
                st.write(weakness)
                
            st.markdown("<h4>💡 Improvement Plan</h4>", unsafe_allow_html=True)
            for suggestion in final_report.get("suggestions", []):
                st.write(f"• {suggestion}")
                
            st.markdown("---")
            
            # Report Download Generation PDF
            report_filename = f"report_{st.session_state.interview_id}.pdf"
            report_filepath = os.path.join("assets/reports", report_filename)
            
            # Draw PDF report via ReportLab doc compilation
            try:
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.pagesizes import letter
                
                doc = SimpleDocTemplate(report_filepath, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
                story = []
                styles = getSampleStyleSheet()
                
                p_style = ParagraphStyle('RepP', parent=styles['Normal'], fontSize=10, leading=14, textColor=colors.HexColor("#334155"))
                h1_style = ParagraphStyle('RepH1', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=18, leading=22, textColor=colors.HexColor("#1e3a8a"), spaceAfter=10)
                h2_style = ParagraphStyle('RepH2', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=12, leading=16, textColor=colors.HexColor("#1d4ed8"), spaceBefore=10, spaceAfter=5)
                
                story.append(Paragraph("AI Mock Interview Performance Report", h1_style))
                story.append(Paragraph(f"Session Candidate: {st.session_state.parsed_resume.get('full_name', 'Student')} | Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", p_style))
                story.append(Spacer(1, 15))
                
                story.append(Paragraph(f"<b>Overall Score: {ov_score}/100</b>", h2_style))
                story.append(Paragraph(f"• Technical Round Score: {final_report.get('technical_score')}%", p_style))
                story.append(Paragraph(f"• HR Round Score: {final_report.get('hr_score')}%", p_style))
                story.append(Paragraph(f"• Verbal Communication: {final_report.get('communication_score')}%", p_style))
                story.append(Paragraph(f"• Non-Verbal Confidence: {final_report.get('confidence_score')}%", p_style))
                story.append(Spacer(1, 10))
                
                story.append(Paragraph("Candidate Strengths", h2_style))
                for s in final_report.get("strengths", []):
                    story.append(Paragraph(s, p_style))
                story.append(Spacer(1, 10))
                
                story.append(Paragraph("Areas of Improvement", h2_style))
                for w in final_report.get("weaknesses", []):
                    story.append(Paragraph(w, p_style))
                story.append(Spacer(1, 10))
                
                story.append(Paragraph("Actionable Recommendations", h2_style))
                for sug in final_report.get("suggestions", []):
                    story.append(Paragraph(f"• {sug}", p_style))
                story.append(Spacer(1, 15))
                
                story.append(Paragraph("Session Interview Q&A Transcript", h2_style))
                for idx, qa in enumerate(st.session_state.interview_transcript):
                    story.append(Paragraph(f"<b>Q{idx+1}: {qa['question']}</b>", p_style))
                    story.append(Paragraph(f"Candidate response: \"{qa['user_answer']}\"", ParagraphStyle('ans', parent=p_style, leftIndent=10, fontName='Helvetica-Oblique')))
                    story.append(Paragraph(f"Evaluation: {qa['feedback']} (Accuracy: {qa['evaluation_accuracy']}% | Conf: {qa['evaluation_confidence']}%)", ParagraphStyle('eval', parent=p_style, leftIndent=10, textColor=colors.HexColor("#475569"))))
                    story.append(Spacer(1, 10))
                    
                doc.build(story)
                
                with open(report_filepath, "rb") as report_file:
                    report_data = report_file.read()
                    
                st.download_button(
                    label="📥 Download Performance Report PDF",
                    data=report_data,
                    file_name=f"Interview_Performance_Report_{st.session_state.interview_id}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as ex:
                st.error(f"Failed to generate report PDF: {ex}")
