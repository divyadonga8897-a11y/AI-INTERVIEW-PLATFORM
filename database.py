import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "placement_advisor.db")

def get_connection():
    """Returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the SQLite database schemas."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Users table (Stores student profile info for Resume Builder/Analyzer)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT,
            email TEXT UNIQUE,
            phone TEXT,
            address TEXT,
            linkedin TEXT,
            github TEXT,
            education TEXT,        -- JSON string of education records
            skills TEXT,           -- JSON list of skills
            projects TEXT,         -- JSON list of projects
            internships TEXT,      -- JSON list of internships
            certifications TEXT,   -- JSON list of certifications
            achievements TEXT,     -- JSON list of achievements
            extracurricular TEXT,  -- JSON list of activities
            objective TEXT,
            languages TEXT,        -- JSON list of languages
            has_photo INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 2. Resume Analysis history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resume_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            file_name TEXT,
            ats_score INTEGER,
            format_score INTEGER,
            skills_score INTEGER,
            projects_score INTEGER,
            experience_score INTEGER,
            keywords_score INTEGER,
            current_skills TEXT,     -- JSON list of current skills
            missing_skills TEXT,     -- JSON list of missing skills
            recommended_courses TEXT, -- JSON list of recommended courses
            suitable_roles TEXT,     -- JSON list of suitable job roles
            weaknesses TEXT,         -- JSON list of weaknesses/warnings
            roadmap TEXT,            -- JSON of 6-month roadmap
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 3. Interview session table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            status TEXT DEFAULT 'IN_PROGRESS', -- 'IN_PROGRESS', 'COMPLETED', 'FAILED'
            overall_score REAL,
            technical_score REAL,
            hr_score REAL,
            communication_score REAL,
            confidence_score REAL,
            strengths TEXT,        -- JSON list of strengths
            weaknesses TEXT,       -- JSON list of weaknesses
            suggestions TEXT,      -- JSON list of recommendations
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 4. Interview questions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interview_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interview_id INTEGER,
            question TEXT,
            round_type TEXT,       -- 'TECHNICAL' or 'HR'
            topic TEXT,            -- e.g., 'Python', 'Machine Learning', 'Self Intro'
            order_index INTEGER,
            FOREIGN KEY (interview_id) REFERENCES interviews(id) ON DELETE CASCADE
        )
    """)
    
    # 5. Interview answers/evaluations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interview_answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id INTEGER,
            user_answer TEXT,
            evaluation_accuracy REAL,
            evaluation_communication REAL,
            evaluation_confidence REAL,
            feedback TEXT,
            fillers_detected TEXT,  -- JSON list of detected fillers
            grammar_issues TEXT,    -- JSON list of grammar issues
            speaking_speed REAL,    -- speed metric (words per min)
            eye_contact_score REAL, -- eye contact metric (0-100)
            expression_summary TEXT, -- e.g., 'Neutral/Confident'
            FOREIGN KEY (question_id) REFERENCES interview_questions(id) ON DELETE CASCADE
        )
    """)
    
    conn.commit()
    conn.close()

# Helper Functions for User Data
def save_user(profile_data):
    """Saves or updates user profile data."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if user already exists
    cursor.execute("SELECT id FROM users WHERE email = ?", (profile_data.get("email"),))
    user = cursor.fetchone()
    
    # Convert list/dict fields to JSON strings
    edu = json.dumps(profile_data.get("education", []))
    skills = json.dumps(profile_data.get("skills", []))
    projects = json.dumps(profile_data.get("projects", []))
    internships = json.dumps(profile_data.get("internships", []))
    certs = json.dumps(profile_data.get("certifications", []))
    achievements = json.dumps(profile_data.get("achievements", []))
    extra = json.dumps(profile_data.get("extracurricular", []))
    languages = json.dumps(profile_data.get("languages", []))
    
    if user:
        cursor.execute("""
            UPDATE users SET
                full_name = ?, phone = ?, address = ?, linkedin = ?, github = ?,
                education = ?, skills = ?, projects = ?, internships = ?,
                certifications = ?, achievements = ?, extracurricular = ?,
                objective = ?, languages = ?, has_photo = ?
            WHERE email = ?
        """, (
            profile_data.get("full_name"), profile_data.get("phone"), profile_data.get("address"),
            profile_data.get("linkedin"), profile_data.get("github"),
            edu, skills, projects, internships, certs, achievements, extra,
            profile_data.get("objective"), languages, 1 if profile_data.get("has_photo") else 0,
            profile_data.get("email")
        ))
        user_id = user["id"]
    else:
        cursor.execute("""
            INSERT INTO users (
                full_name, email, phone, address, linkedin, github,
                education, skills, projects, internships, certifications,
                achievements, extracurricular, objective, languages, has_photo
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            profile_data.get("full_name"), profile_data.get("email"), profile_data.get("phone"),
            profile_data.get("address"), profile_data.get("linkedin"), profile_data.get("github"),
            edu, skills, projects, internships, certs, achievements, extra,
            profile_data.get("objective"), languages, 1 if profile_data.get("has_photo") else 0
        ))
        user_id = cursor.lastrowid
        
    conn.commit()
    conn.close()
    return user_id

def get_user_by_email(email):
    """Retrieves user profile data by email, loaded back into objects from JSON strings."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
        
    user_dict = dict(row)
    # Parse JSON strings back to lists/dicts
    for key in ["education", "skills", "projects", "internships", "certifications", "achievements", "extracurricular", "languages"]:
        if user_dict[key]:
            user_dict[key] = json.loads(user_dict[key])
        else:
            user_dict[key] = []
            
    return user_dict

# Helper Functions for Resume Analysis
def save_resume_analysis(analysis_data):
    """Saves a resume analysis record."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO resume_analysis (
            email, file_name, ats_score, format_score, skills_score,
            projects_score, experience_score, keywords_score,
            current_skills, missing_skills, recommended_courses,
            suitable_roles, weaknesses, roadmap
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        analysis_data.get("email"),
        analysis_data.get("file_name"),
        analysis_data.get("ats_score"),
        analysis_data.get("format_score"),
        analysis_data.get("skills_score"),
        analysis_data.get("projects_score"),
        analysis_data.get("experience_score"),
        analysis_data.get("keywords_score"),
        json.dumps(analysis_data.get("current_skills", [])),
        json.dumps(analysis_data.get("missing_skills", [])),
        json.dumps(analysis_data.get("recommended_courses", [])),
        json.dumps(analysis_data.get("suitable_roles", [])),
        json.dumps(analysis_data.get("weaknesses", [])),
        json.dumps(analysis_data.get("roadmap", {}))
    ))
    conn.commit()
    conn.close()

def get_latest_resume_analysis(email):
    """Gets the most recent resume analysis report for an email."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM resume_analysis 
        WHERE email = ? 
        ORDER BY created_at DESC LIMIT 1
    """, (email,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
        
    report = dict(row)
    for key in ["current_skills", "missing_skills", "recommended_courses", "suitable_roles", "weaknesses"]:
        report[key] = json.loads(report[key]) if report[key] else []
    report["roadmap"] = json.loads(report["roadmap"]) if report["roadmap"] else {}
    return report

# Helper Functions for Interview Sessions
def create_interview(email):
    """Creates a new interview record and returns its ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO interviews (email) VALUES (?)", (email,))
    interview_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return interview_id

def update_interview(interview_id, details):
    """Updates interview evaluation fields when completed."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE interviews SET
            status = ?,
            overall_score = ?,
            technical_score = ?,
            hr_score = ?,
            communication_score = ?,
            confidence_score = ?,
            strengths = ?,
            weaknesses = ?,
            suggestions = ?
        WHERE id = ?
    """, (
        details.get("status", "COMPLETED"),
        details.get("overall_score", 0.0),
        details.get("technical_score", 0.0),
        details.get("hr_score", 0.0),
        details.get("communication_score", 0.0),
        details.get("confidence_score", 0.0),
        json.dumps(details.get("strengths", [])),
        json.dumps(details.get("weaknesses", [])),
        json.dumps(details.get("suggestions", [])),
        interview_id
    ))
    conn.commit()
    conn.close()

def add_interview_question(interview_id, question, round_type, topic, order_index):
    """Saves an interview question."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO interview_questions (interview_id, question, round_type, topic, order_index)
        VALUES (?, ?, ?, ?, ?)
    """, (interview_id, question, round_type, topic, order_index))
    q_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return q_id

def save_interview_answer(question_id, answer_details):
    """Saves user's response and its granular evaluation metrics."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO interview_answers (
            question_id, user_answer, evaluation_accuracy, 
            evaluation_communication, evaluation_confidence, feedback,
            fillers_detected, grammar_issues, speaking_speed, eye_contact_score, expression_summary
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        question_id,
        answer_details.get("user_answer", ""),
        answer_details.get("evaluation_accuracy", 0.0),
        answer_details.get("evaluation_communication", 0.0),
        answer_details.get("evaluation_confidence", 0.0),
        answer_details.get("feedback", ""),
        json.dumps(answer_details.get("fillers_detected", [])),
        json.dumps(answer_details.get("grammar_issues", [])),
        answer_details.get("speaking_speed", 120.0),
        answer_details.get("eye_contact_score", 100.0),
        answer_details.get("expression_summary", "Neutral")
    ))
    conn.commit()
    conn.close()

def get_interview_transcript(interview_id):
    """Retrieves all questions and their corresponding answers for an interview session."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT q.id as q_id, q.question, q.round_type, q.topic, q.order_index,
               a.user_answer, a.evaluation_accuracy, a.evaluation_communication, 
               a.evaluation_confidence, a.feedback, a.fillers_detected, 
               a.grammar_issues, a.speaking_speed, a.eye_contact_score, a.expression_summary
        FROM interview_questions q
        LEFT JOIN interview_answers a ON q.id = a.question_id
        WHERE q.interview_id = ?
        ORDER BY q.order_index ASC
    """, (interview_id,))
    rows = cursor.fetchall()
    conn.close()
    
    transcript = []
    for row in rows:
        item = dict(row)
        if item["fillers_detected"]:
            item["fillers_detected"] = json.loads(item["fillers_detected"])
        else:
            item["fillers_detected"] = []
        if item["grammar_issues"]:
            item["grammar_issues"] = json.loads(item["grammar_issues"])
        else:
            item["grammar_issues"] = []
        transcript.append(item)
    return transcript

def get_interview_by_id(interview_id):
    """Retrieves an interview summary record by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM interviews WHERE id = ?", (interview_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
        
    interview = dict(row)
    interview["strengths"] = json.loads(interview["strengths"]) if interview["strengths"] else []
    interview["weaknesses"] = json.loads(interview["weaknesses"]) if interview["weaknesses"] else []
    interview["suggestions"] = json.loads(interview["suggestions"]) if interview["suggestions"] else []
    return interview

def get_user_interviews_history(email):
    """Retrieves list of past interviews for a user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM interviews 
        WHERE email = ? 
        ORDER BY created_at DESC
    """, (email,))
    rows = cursor.fetchall()
    conn.close()
    
    interviews = []
    for r in rows:
        item = dict(r)
        item["strengths"] = json.loads(item["strengths"]) if item["strengths"] else []
        item["weaknesses"] = json.loads(item["weaknesses"]) if item["weaknesses"] else []
        item["suggestions"] = json.loads(item["suggestions"]) if item["suggestions"] else []
        interviews.append(item)
    return interviews
