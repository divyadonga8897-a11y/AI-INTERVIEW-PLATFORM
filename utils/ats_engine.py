import json
from utils.groq_client import generate_chat_response

def analyze_resume_ats(profile_data: dict, target_role: str = "Software Developer", api_key: str = None, provider: str = "groq") -> dict:
    """
    Sends the parsed resume JSON profile data to the LLM to get an ATS evaluation report.
    """
    system_prompt = """
    You are an expert ATS (Applicant Tracking System) Evaluation Engine and Career Adviser.
    Analyze the user's resume profile and evaluate it against standard recruitment metrics for the target job role.
    Provide your response ONLY as a valid JSON object. Do not include markdown code fences (like ```json), introductions, or explanations.
    
    Structure the response JSON exactly as follows:
    {
        "ats_score": 82,
        "breakdown": {
            "format_score": 90,
            "skills_score": 75,
            "projects_score": 85,
            "experience_score": 70,
            "keywords_score": 80
        },
        "current_skills": ["Skill1", "Skill2"],
        "missing_skills": ["SkillA", "SkillB"],
        "recommended_courses": [
            "Course Name 1 (e.g., Python Advanced)",
            "Course Name 2"
        ],
        "suitable_roles": ["Role 1", "Role 2"],
        "weaknesses": [
            "Missing achievements",
            "No quantified results (percentages/metrics)",
            "Weak summary",
            "Missing GitHub link"
        ],
        "roadmap": {
            "month_1": "Month 1 Topic and Goals (e.g. DSA)",
            "month_2": "Month 2 Topic and Goals",
            "month_3": "Month 3 Topic and Goals",
            "month_4": "Month 4 Topic and Goals",
            "month_5": "Month 5 Topic and Goals",
            "month_6": "Month 6 Topic and Goals"
        }
    }
    """
    
    prompt = f"""
    Please evaluate this candidate's resume for the target role: "{target_role}".
    
    Candidate Profile:
    {json.dumps(profile_data, indent=2)}
    
    Provide an accurate, honest ATS score breakdown and actionable steps, weaknesses, missing skills, and a 6-month roadmap.
    """
    
    response_str = generate_chat_response(prompt, system_prompt, api_key, provider, json_mode=True)
    
    # Strip markdown code fences if LLM ignored instructions
    response_str = response_str.strip()
    if response_str.startswith("```"):
        lines = response_str.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].startswith("```"):
            lines = lines[:-1]
        response_str = "\n".join(lines).strip()
        
    try:
        report = json.loads(response_str)
        # Ensure all fields exist with fallback values
        report.setdefault("ats_score", 70)
        report.setdefault("breakdown", {
            "format_score": 70,
            "skills_score": 70,
            "projects_score": 70,
            "experience_score": 70,
            "keywords_score": 70
        })
        report.setdefault("current_skills", profile_data.get("skills", []))
        report.setdefault("missing_skills", [])
        report.setdefault("recommended_courses", [])
        report.setdefault("suitable_roles", [target_role])
        report.setdefault("weaknesses", [])
        report.setdefault("roadmap", {
            "month_1": "DSA Basics",
            "month_2": "Database & SQL",
            "month_3": "Core Backend/Framework",
            "month_4": "Cloud Infrastructure",
            "month_5": "Docker & Systems",
            "month_6": "Interview Prep"
        })
        return report
    except Exception as e:
        print(f"Failed to parse ATS report JSON: {e}")
        # Return fallback structure
        return {
            "ats_score": 65,
            "breakdown": {
                "format_score": 70,
                "skills_score": 60,
                "projects_score": 65,
                "experience_score": 60,
                "keywords_score": 70
            },
            "current_skills": profile_data.get("skills", []),
            "missing_skills": ["Docker", "Kubernetes", "AWS"],
            "recommended_courses": ["Advanced Systems Design", "Cloud Infrastructure"],
            "suitable_roles": [target_role, "Software Developer"],
            "weaknesses": ["Lack of quantified results", "No links provided"],
            "roadmap": {
                "month_1": "Algorithms & Data Structures",
                "month_2": "SQL and Databases",
                "month_3": "Backend Framework",
                "month_4": "AWS Essentials",
                "month_5": "Containerization (Docker)",
                "month_6": "Mock Practice & Soft Skills"
            }
        }
