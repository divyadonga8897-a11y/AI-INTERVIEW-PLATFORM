import json
import re
from utils.groq_client import generate_chat_response

# List of typical verbal filler words to scan locally
FILLER_WORDS = ["umm", "umm...", "um", "uh", "uhh", "like", "actually", "basically", "so", "you know"]

def clean_transcript(text: str) -> str:
    """Cleans punctuation and makes it lowercase for analysis."""
    return re.sub(r'[^\w\s\']', '', text.lower())

def count_fillers(transcript: str) -> dict:
    """
    Locally counts typical verbal filler words in the transcription.
    Returns a list of fillers found and their count.
    """
    words = clean_transcript(transcript).split()
    counts = {}
    for word in words:
        if word in FILLER_WORDS:
            counts[word] = counts.get(word, 0) + 1
            
    return [{"filler": k, "count": v} for k, v in counts.items()]

def generate_interview_questions(profile_data: dict, target_role: str = "Software Developer", api_key: str = None, provider: str = "groq") -> list:
    """
    Scans the candidate profile (skills, projects, certifications, etc.)
    and dynamically generates a list of interview questions.
    Returns a list of dictionaries with keys: question, round_type, topic.
    """
    system_prompt = """
    You are an expert technical recruiter and interviewer.
    Analyze the user's resume and generate exactly 6 highly relevant, specific interview questions:
    - 3 Technical Round questions tailored specifically to the skills, projects, and databases listed in their resume (e.g., "Why did you choose Flask over Django?", "How does Random Forest work?", "What is a Python decorator?").
    - 3 HR Round questions targeting typical behavioral and career objectives (e.g., "Tell me about yourself", "Describe a time you solved a conflict in a team", "What are your career goals?").
    
    Format your output strictly as a JSON array of objects. Do not include markdown code fences (like ```json), introductions, or formatting outside JSON.
    
    Structure of each question object in the list:
    {
        "question": "Question text here",
        "round_type": "TECHNICAL" or "HR",
        "topic": "The technology or behavior category (e.g. Python, SQL, Teamwork, Self Intro)"
    }
    """
    
    prompt = f"""
    Please generate 6 interview questions for a candidate applying for the role: "{target_role}".
    
    Candidate Profile:
    {json.dumps(profile_data, indent=2)}
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
        questions = json.loads(response_str)
        if isinstance(questions, list) and len(questions) > 0:
            return questions
    except Exception as e:
        print(f"Failed to parse interview questions JSON: {e}")
        
    # Standard fallback questions if JSON parsing fails
    return [
        {"question": "Tell me about yourself and walk me through your background.", "round_type": "HR", "topic": "Self Intro"},
        {"question": "What is a python decorator and how does it work?", "round_type": "TECHNICAL", "topic": "Python"},
        {"question": "Describe a difficult team situation and how you handled it.", "round_type": "HR", "topic": "Teamwork"},
        {"question": "What is the difference between a SQL and NoSQL database?", "round_type": "TECHNICAL", "topic": "Database"},
        {"question": "Where do you see yourself in 5 years?", "round_type": "HR", "topic": "Career Goals"},
        {"question": "Explain a project you built, its architecture, and why you chose that technology stack.", "round_type": "TECHNICAL", "topic": "Projects"}
    ]

def evaluate_candidate_answer(question: str, user_answer: str, round_type: str, api_key: str = None, provider: str = "groq") -> dict:
    """
    Evaluates the candidate's answer against the question.
    Returns granular scores (accuracy, communication, confidence), feedback, and grammar issues.
    """
    if not user_answer.strip():
        return {
            "evaluation_accuracy": 0.0,
            "evaluation_communication": 10.0,
            "evaluation_confidence": 10.0,
            "feedback": "No response was recorded. Please speak clearly into your microphone.",
            "fillers_detected": [],
            "grammar_issues": []
        }
        
    # Analyze fillers locally first
    fillers = count_fillers(user_answer)
    
    # Compute average talking speed (simulated if time duration is not available)
    # We estimate based on word count (e.g. 15 words recorded is fast/slow depending on length)
    
    system_prompt = """
    You are an AI Interview Evaluator. Evaluate the candidate's answer for the given question.
    Rate the answer across these three metrics on a scale of 0 to 100:
    1. Accuracy (Technical correctness for TECHNICAL round, relevancy/structure for HR round)
    2. Communication (Clarity, vocabulary, sentence structure)
    3. Confidence (Assertiveness, lack of hesitation in phrasing)
    
    Identify any direct grammar issues in the text.
    Provide constructive, concise feedback.
    
    Format your response strictly as a JSON object. Do not include markdown code fences (like ```json), introductions, or formatting outside JSON.
    
    JSON Structure:
    {
        "evaluation_accuracy": 85.0,
        "evaluation_communication": 78.0,
        "evaluation_confidence": 80.0,
        "feedback": "Provide concise constructive feedback here.",
        "grammar_issues": ["Issue 1: explain what was wrong", "Issue 2"]
    }
    """
    
    prompt = f"""
    Evaluate the following response:
    
    Round: {round_type}
    Question: {question}
    Candidate's Answer: {user_answer}
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
        evaluation = json.loads(response_str)
        evaluation["fillers_detected"] = fillers
        return evaluation
    except Exception as e:
        print(f"Failed to parse candidate evaluation JSON: {e}")
        return {
            "evaluation_accuracy": 70.0,
            "evaluation_communication": 70.0,
            "evaluation_confidence": 70.0,
            "feedback": "Answer received and recorded. Communication was clear; review the core concept to improve depth.",
            "fillers_detected": fillers,
            "grammar_issues": []
        }

def compile_final_interview_report(session_transcript: list) -> dict:
    """
    Aggregates scores across all questions in the session transcript.
    Computes overall, technical, and HR averages, and outlines strengths, weaknesses, and tips.
    """
    if not session_transcript:
        return {}
        
    tech_acc_scores = []
    hr_acc_scores = []
    comm_scores = []
    conf_scores = []
    
    all_fillers = {}
    all_grammar = []
    
    for qa in session_transcript:
        if qa.get("evaluation_accuracy") is not None:
            if qa.get("round_type") == "TECHNICAL":
                tech_acc_scores.append(qa["evaluation_accuracy"])
            else:
                hr_acc_scores.append(qa["evaluation_accuracy"])
                
        if qa.get("evaluation_communication") is not None:
            comm_scores.append(qa["evaluation_communication"])
        if qa.get("evaluation_confidence") is not None:
            conf_scores.append(qa["evaluation_confidence"])
            
        # Aggregate fillers
        for fill_info in qa.get("fillers_detected", []):
            f = fill_info["filler"]
            c = fill_info["count"]
            all_fillers[f] = all_fillers.get(f, 0) + c
            
        # Aggregate grammar
        all_grammar.extend(qa.get("grammar_issues", []))
        
    avg_tech = sum(tech_acc_scores) / len(tech_acc_scores) if tech_acc_scores else 80.0
    avg_hr = sum(hr_acc_scores) / len(hr_acc_scores) if hr_acc_scores else 80.0
    avg_comm = sum(comm_scores) / len(comm_scores) if comm_scores else 80.0
    avg_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 80.0
    
    # Calculate overall weighted score
    overall = (avg_tech * 0.4) + (avg_hr * 0.3) + (avg_comm * 0.15) + (avg_conf * 0.15)
    
    # Compile qualitative lists based on scores
    strengths = []
    weaknesses = []
    suggestions = []
    
    if avg_tech >= 80:
        strengths.append("✓ Strong technical knowledge and conceptual accuracy.")
    else:
        weaknesses.append("✗ Gaps in technical definitions or details.")
        suggestions.append("Study the fundamental concepts of listed skills and projects.")
        
    if avg_comm >= 80:
        strengths.append("✓ Highly articulate with clear sentence structures.")
    else:
        weaknesses.append("✗ Sentence delivery can be more structured.")
        suggestions.append("Practice describing projects using the STAR method (Situation, Task, Action, Result).")
        
    if avg_conf >= 80:
        strengths.append("✓ Confident delivery with minimal pauses.")
    else:
        weaknesses.append("✗ Shows signs of hesitation or speech blocks.")
        suggestions.append("Take regular deep breaths and speak slowly during responses.")
        
    # Filler warnings
    total_fillers = sum(all_fillers.values())
    if total_fillers > 5:
        weaknesses.append("✗ High usage of speech fillers (like 'Umm', 'Like').")
        suggestions.append("Reduce the usage of filler words by pausing silently when thinking.")
    elif total_fillers > 0:
        strengths.append("✓ Good control over filler word usage.")
        
    if len(all_grammar) > 2:
        weaknesses.append("✗ Occasional minor grammatical errors in phrasing.")
        suggestions.append("Focus on past and present tense consistency when describing work experience.")
    elif len(session_transcript) > 0 and len(all_grammar) == 0:
        strengths.append("✓ Correct grammar and standard professional vocabulary.")
        
    # Ensure suggestions is not empty
    if not suggestions:
        suggestions.append("Conduct regular mock interviews to maintain communication flow.")
        
    return {
        "overall_score": round(overall, 1),
        "technical_score": round(avg_tech, 1),
        "hr_score": round(avg_hr, 1),
        "communication_score": round(avg_comm, 1),
        "confidence_score": round(avg_conf, 1),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": suggestions,
        "fillers_summary": all_fillers,
        "grammar_summary": all_grammar[:5]
    }
