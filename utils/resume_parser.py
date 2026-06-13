import PyPDF2
import os

def extract_text_from_pdf(pdf_file_path_or_obj) -> str:
    """
    Extracts text from a PDF file path or file-like object.
    Attempts to use pdfplumber for better extraction, falls back to PyPDF2.
    """
    text = ""
    
    # Try pdfplumber first
    try:
        import pdfplumber
        if isinstance(pdf_file_path_or_obj, str):
            with pdfplumber.open(pdf_file_path_or_obj) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        else:
            with pdfplumber.open(pdf_file_path_or_obj) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        
        # If text is extracted successfully, return it
        if text.strip():
            return text.strip()
    except Exception as e:
        print(f"pdfplumber extraction failed or not available, trying PyPDF2. Error: {e}")

    # Fallback to PyPDF2
    try:
        if isinstance(pdf_file_path_or_obj, str):
            with open(pdf_file_path_or_obj, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        else:
            # It's already a file-like object (e.g. BytesIO from streamlit file_uploader)
            # Make sure we seek to 0 in case it was read
            try:
                pdf_file_path_or_obj.seek(0)
            except Exception:
                pass
            reader = PyPDF2.PdfReader(pdf_file_path_or_obj)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    
        return text.strip()
    except Exception as e:
        print(f"PyPDF2 extraction failed. Error: {e}")
        return ""

def parse_resume_to_json(raw_text: str, api_key: str = None, provider: str = "groq") -> dict:
    """
    Uses the LLM to structure raw resume text into a standard JSON format.
    """
    if not raw_text.strip():
        return {}
        
    system_prompt = """
    You are an expert AI Resume Parser. Analyze the raw text of a resume and extract the details into a structured JSON format.
    Ensure that you return ONLY a valid JSON object. Do not include markdown code fences (like ```json), introductions, or formatting outside JSON.
    
    Structure the JSON exactly as follows:
    {
        "full_name": "Full Name",
        "email": "Email Address",
        "phone": "Phone Number",
        "address": "Address or Location",
        "linkedin": "LinkedIn profile link (empty if not found)",
        "github": "GitHub profile link (empty if not found)",
        "objective": "Professional summary or career objective",
        "education": [
            {
                "institution": "College/School Name",
                "degree": "Degree/Major",
                "cgpa": "CGPA/Percentage or Grade"
            }
        ],
        "skills": ["Skill1", "Skill2"],
        "projects": [
            {
                "title": "Project Name",
                "description": "Short description of what was done",
                "technologies": ["Tech1", "Tech2"]
            }
        ],
        "internships": [
            {
                "company": "Company Name",
                "role": "Internship Role",
                "description": "Short description of duties/achievements",
                "duration": "Duration (e.g., 3 months)"
            }
        ],
        "certifications": ["Cert1", "Cert2"],
        "achievements": ["Achievement1", "Achievement2"],
        "extracurricular": ["Activity1", "Activity2"],
        "languages": ["Lang1", "Lang2"]
    }
    """
    
    prompt = f"Extract information from this raw resume text:\n\n{raw_text}"
    
    from utils.groq_client import generate_chat_response
    import json
    
    response_str = generate_chat_response(prompt, system_prompt, api_key, provider, json_mode=True)
    
    # Strip markdown code fences if LLM ignored instructions
    response_str = response_str.strip()
    if response_str.startswith("```"):
        # Remove first line
        lines = response_str.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].startswith("```"):
            lines = lines[:-1]
        response_str = "\n".join(lines).strip()
        
    try:
        parsed_data = json.loads(response_str)
        return parsed_data
    except Exception as e:
        print(f"Failed to parse LLM response to JSON: {e}")
        print("Raw response was:", response_str)
        # Attempt simple JSON recovery or return placeholder
        return {
            "full_name": "Unknown Candidate",
            "email": "",
            "phone": "",
            "skills": [],
            "education": [],
            "projects": [],
            "raw_text": raw_text[:500]
        }
