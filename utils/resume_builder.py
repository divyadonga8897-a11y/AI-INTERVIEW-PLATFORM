import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from utils.groq_client import generate_chat_response

def enhance_bullet_point(text: str, api_key: str = None, provider: str = "groq") -> str:
    """
    Polishes a raw resume bullet point using AI.
    Example: 'Developed a website' -> 'Designed and developed a responsive web application...'
    """
    if not text.strip():
        return ""
        
    system_prompt = """
    You are an expert resume writer. Transform the user's simple work description or project bullet point into a highly professional, result-oriented, and action-verb-driven resume achievement.
    Use active verbs, describe the technologies used, and where possible, include simulated quantified metrics (like improvement percentages or efficiency metrics).
    Keep it concise—exactly one strong sentence. Return ONLY the enhanced sentence. Do not include quotes, introductions, or explanations.
    """
    
    prompt = f"Original: {text}"
    enhanced = generate_chat_response(prompt, system_prompt, api_key, provider)
    # Strip quotes if the LLM returned them
    return enhanced.strip().strip('"').strip("'")


def generate_pdf_resume(profile: dict, template_name: str, output_path: str):
    """
    Generates a beautifully typeset PDF resume using reportlab.
    Templates: 'Modern', 'Professional', 'ATS Friendly', 'Creative'.
    """
    # 1. Setup Document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    # 2. Get Styles
    styles = getSampleStyleSheet()
    
    # Colors depending on template
    primary_color = colors.HexColor("#1e293b")  # default charcoal/slate
    secondary_color = colors.HexColor("#475569")
    text_color = colors.HexColor("#334155")
    bg_color = colors.HexColor("#ffffff")
    accent_color = colors.HexColor("#0284c7")
    
    if template_name == "Professional":
        primary_color = colors.HexColor("#1e3a8a")  # Navy Blue
        secondary_color = colors.HexColor("#1d4ed8")
        accent_color = colors.HexColor("#3b82f6")
    elif template_name == "ATS Friendly":
        primary_color = colors.HexColor("#000000")  # Strict black and white
        secondary_color = colors.HexColor("#262626")
        accent_color = colors.HexColor("#404040")
        text_color = colors.HexColor("#000000")
    elif template_name == "Creative":
        primary_color = colors.HexColor("#581c87")  # Deep Purple
        secondary_color = colors.HexColor("#7e22ce")
        accent_color = colors.HexColor("#a855f7")
        bg_color = colors.HexColor("#faf5ff")
    elif template_name == "Modern":
        primary_color = colors.HexColor("#0f172a")  # Slate-900
        secondary_color = colors.HexColor("#0d9488") # Teal
        accent_color = colors.HexColor("#14b8a6")
        
    # Custom Paragraph Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=primary_color,
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=secondary_color,
        spaceAfter=10
    )
    
    h1_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=primary_color,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'SubSectionHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=secondary_color,
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=text_color,
        spaceAfter=3
    )
    
    bullet_style = ParagraphStyle(
        'BulletCustom',
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=2
    )

    story = []
    
    # ------------------ TEMPLATE RENDERERS ------------------
    
    # Helper to construct sections
    def get_contact_info_text():
        email = profile.get('email', '')
        phone = profile.get('phone', '')
        address = profile.get('address', '')
        linkedin = profile.get('linkedin', '')
        github = profile.get('github', '')
        
        info = []
        if email: info.append(email)
        if phone: info.append(phone)
        if address: info.append(address)
        if linkedin: info.append(f"LinkedIn: {linkedin}")
        if github: info.append(f"GitHub: {github}")
        
        return "  |  ".join(info)

    def draw_section_divider(title_text):
        # Creates a heading and a horizontal line
        heading = Paragraph(title_text.upper(), h1_style)
        
        # Horizontal line can be drawn using a table with border
        line = Table([[""]], colWidths=[540])
        line.setStyle(TableStyle([
            ('LINEABOVE', (0,0), (-1,-1), 1, primary_color),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        return [heading, line, Spacer(1, 4)]

    # General Single-Column Builder (ATS / Professional / Creative)
    if template_name != "Modern":
        # Header (Name)
        story.append(Paragraph(profile.get('full_name', 'Your Name'), title_style))
        story.append(Paragraph(get_contact_info_text(), subtitle_style))
        story.append(Spacer(1, 10))
        
        # Objective
        if profile.get('objective'):
            story.extend(draw_section_divider("Career Objective"))
            story.append(Paragraph(profile.get('objective'), body_style))
            story.append(Spacer(1, 8))
            
        # Education
        if profile.get('education'):
            story.extend(draw_section_divider("Education"))
            for edu in profile['education']:
                degree_text = f"<b>{edu.get('degree', '')}</b>"
                cgpa = edu.get('cgpa', '')
                if cgpa:
                    degree_text += f" (CGPA: {cgpa})"
                
                edu_table_data = [
                    [Paragraph(degree_text, body_style), Paragraph(f"<font color='{secondary_color.hexval()}'><b>{edu.get('institution', '')}</b></font>", body_style)]
                ]
                edu_table = Table(edu_table_data, colWidths=[300, 240])
                edu_table.setStyle(TableStyle([
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ('RIGHTPADDING', (0,0), (-1,-1), 0),
                    ('LEFTPADDING', (0,0), (-1,-1), 0),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 2),
                    ('TOPPADDING', (0,0), (-1,-1), 2),
                ]))
                story.append(edu_table)
            story.append(Spacer(1, 8))
            
        # Technical Skills
        if profile.get('skills'):
            story.extend(draw_section_divider("Technical Skills"))
            skills_list = profile['skills']
            if isinstance(skills_list, list):
                skills_str = ", ".join(skills_list)
            else:
                skills_str = str(skills_list)
            story.append(Paragraph(skills_str, body_style))
            story.append(Spacer(1, 8))
            
        # Experience / Internships
        if profile.get('internships') or profile.get('experience'):
            story.extend(draw_section_divider("Work Experience & Internships"))
            jobs = profile.get('internships', []) + profile.get('experience', [])
            for job in jobs:
                role_text = f"<b>{job.get('role', job.get('title', 'Intern'))}</b> - {job.get('company', job.get('organization', ''))}"
                duration = job.get('duration', job.get('date', ''))
                
                header_table = Table([
                    [Paragraph(role_text, h2_style), Paragraph(f"<i>{duration}</i>", ParagraphStyle('dur', parent=body_style, alignment=2))]
                ], colWidths=[400, 140])
                header_table.setStyle(TableStyle([
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ('LEFTPADDING', (0,0), (-1,-1), 0),
                    ('RIGHTPADDING', (0,0), (-1,-1), 0),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 1),
                    ('TOPPADDING', (0,0), (-1,-1), 1),
                ]))
                story.append(header_table)
                
                desc = job.get('description', '')
                if desc:
                    if "\n" in desc or desc.startswith("-") or desc.startswith("•"):
                        # Bulleted description
                        lines = [line.strip().lstrip("-• ") for line in desc.split("\n") if line.strip()]
                        for line in lines:
                            story.append(Paragraph(f"• {line}", bullet_style))
                    else:
                        story.append(Paragraph(desc, body_style))
                story.append(Spacer(1, 4))
            story.append(Spacer(1, 6))
            
        # Projects
        if profile.get('projects'):
            story.extend(draw_section_divider("Key Projects"))
            for proj in profile['projects']:
                title = proj.get('title', '')
                techs = proj.get('technologies', [])
                tech_str = f" [Tech: {', '.join(techs)}]" if techs else ""
                
                story.append(Paragraph(f"<b>{title}</b>{tech_str}", h2_style))
                
                desc = proj.get('description', '')
                if desc:
                    if "\n" in desc or desc.startswith("-") or desc.startswith("•"):
                        lines = [line.strip().lstrip("-• ") for line in desc.split("\n") if line.strip()]
                        for line in lines:
                            story.append(Paragraph(f"• {line}", bullet_style))
                    else:
                        story.append(Paragraph(desc, body_style))
                story.append(Spacer(1, 4))
            story.append(Spacer(1, 6))
            
        # Certifications & Achievements
        certs = profile.get('certifications', [])
        achievements = profile.get('achievements', [])
        if certs or achievements:
            story.extend(draw_section_divider("Certifications & Achievements"))
            for c in certs:
                story.append(Paragraph(f"• {c}", bullet_style))
            for a in achievements:
                story.append(Paragraph(f"• {a}", bullet_style))
            story.append(Spacer(1, 8))
            
    else:
        # Modern: Left Sidebar + Right Main Column
        # We model this structure in reportlab using a master table: 1 row, 2 columns.
        # Column 1 (Left): 170 points width. Column 2 (Right): 370 points width.
        # Total printable width is 540 (letter width is 612 - 72 margins)
        
        # Prepare Sidebar contents
        sidebar_flow = []
        sidebar_flow.append(Paragraph(profile.get('full_name', 'Your Name'), title_style))
        sidebar_flow.append(Spacer(1, 10))
        
        sidebar_title_style = ParagraphStyle('SidebarHead', parent=h1_style, textColor=colors.white if bg_color != colors.white else primary_color, fontSize=11, leading=14)
        sidebar_body_style = ParagraphStyle('SidebarBody', parent=body_style, textColor=colors.HexColor("#e2e8f0") if bg_color != colors.white else text_color, fontSize=8, leading=11)
        
        # Contact Details
        sidebar_flow.append(Paragraph("<b>CONTACT</b>", sidebar_title_style))
        sidebar_flow.append(Spacer(1, 2))
        for key in ['email', 'phone', 'address', 'linkedin', 'github']:
            val = profile.get(key, '')
            if val:
                sidebar_flow.append(Paragraph(f"<b>{key.upper()}</b>:<br/>{val}", sidebar_body_style))
                sidebar_flow.append(Spacer(1, 4))
        
        sidebar_flow.append(Spacer(1, 15))
        
        # Skills
        if profile.get('skills'):
            sidebar_flow.append(Paragraph("<b>SKILLS</b>", sidebar_title_style))
            sidebar_flow.append(Spacer(1, 4))
            for s in profile['skills']:
                sidebar_flow.append(Paragraph(f"• {s}", sidebar_body_style))
            sidebar_flow.append(Spacer(1, 15))
            
        # Languages
        if profile.get('languages'):
            sidebar_flow.append(Paragraph("<b>LANGUAGES</b>", sidebar_title_style))
            sidebar_flow.append(Spacer(1, 4))
            for l in profile['languages']:
                sidebar_flow.append(Paragraph(f"• {l}", sidebar_body_style))
        
        # Prepare Main Column contents
        main_flow = []
        
        # Objective
        if profile.get('objective'):
            main_flow.append(Paragraph("<b>OBJECTIVE</b>", h1_style))
            main_flow.append(Paragraph(profile.get('objective'), body_style))
            main_flow.append(Spacer(1, 10))
            
        # Education
        if profile.get('education'):
            main_flow.append(Paragraph("<b>EDUCATION</b>", h1_style))
            for edu in profile['education']:
                deg = edu.get('degree', '')
                inst = edu.get('institution', '')
                cgpa = edu.get('cgpa', '')
                cgpa_str = f" | CGPA: {cgpa}" if cgpa else ""
                main_flow.append(Paragraph(f"<b>{deg}</b>{cgpa_str}", h2_style))
                main_flow.append(Paragraph(inst, body_style))
                main_flow.append(Spacer(1, 5))
            main_flow.append(Spacer(1, 10))
            
        # Internships / Experience
        jobs = profile.get('internships', []) + profile.get('experience', [])
        if jobs:
            main_flow.append(Paragraph("<b>EXPERIENCE</b>", h1_style))
            for job in jobs:
                role = job.get('role', job.get('title', ''))
                comp = job.get('company', job.get('organization', ''))
                dur = job.get('duration', job.get('date', ''))
                
                main_flow.append(Paragraph(f"<b>{role}</b> - {comp}", h2_style))
                main_flow.append(Paragraph(f"<i>{dur}</i>", ParagraphStyle('ital', parent=body_style, fontSize=8)))
                
                desc = job.get('description', '')
                if desc:
                    if "\n" in desc or desc.startswith("-") or desc.startswith("•"):
                        lines = [line.strip().lstrip("-• ") for line in desc.split("\n") if line.strip()]
                        for line in lines:
                            main_flow.append(Paragraph(f"• {line}", bullet_style))
                    else:
                        main_flow.append(Paragraph(desc, body_style))
                main_flow.append(Spacer(1, 5))
            main_flow.append(Spacer(1, 10))
            
        # Projects
        if profile.get('projects'):
            main_flow.append(Paragraph("<b>PROJECTS</b>", h1_style))
            for proj in profile['projects']:
                title = proj.get('title', '')
                techs = proj.get('technologies', [])
                tech_str = f" ({', '.join(techs)})" if techs else ""
                
                main_flow.append(Paragraph(f"<b>{title}</b>{tech_str}", h2_style))
                
                desc = proj.get('description', '')
                if desc:
                    if "\n" in desc or desc.startswith("-") or desc.startswith("•"):
                        lines = [line.strip().lstrip("-• ") for line in desc.split("\n") if line.strip()]
                        for line in lines:
                            main_flow.append(Paragraph(f"• {line}", bullet_style))
                    else:
                        main_flow.append(Paragraph(desc, body_style))
                main_flow.append(Spacer(1, 5))
            main_flow.append(Spacer(1, 10))

        # Certifications & Achievements
        certs = profile.get('certifications', [])
        achievements = profile.get('achievements', [])
        if certs or achievements:
            main_flow.append(Paragraph("<b>ACHIEVEMENTS</b>", h1_style))
            for c in certs:
                main_flow.append(Paragraph(f"• {c}", bullet_style))
            for a in achievements:
                main_flow.append(Paragraph(f"• {a}", bullet_style))
        
        # Assemble Master Table
        # We put sidebar and main column inside a single row of a table
        # Since reportlab flowables need to be inside cells, we can do this!
        master_table_data = [[sidebar_flow, main_flow]]
        master_table = Table(master_table_data, colWidths=[175, 365])
        
        # Design background of sidebar cell
        # Column 0 is sidebar (charcoal background slate-900), Column 1 is main flow (white)
        master_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,0), primary_color),
            ('BACKGROUND', (1,0), (1,0), colors.white),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ('LEFTPADDING', (0,0), (0,0), 12),
            ('RIGHTPADDING', (0,0), (0,0), 12),
            ('LEFTPADDING', (1,0), (1,0), 15),
            ('RIGHTPADDING', (1,0), (1,0), 0),
        ]))
        story.append(master_table)
        
    # Build Document
    doc.build(story)
