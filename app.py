from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

from flask import Flask, flash, jsonify, make_response, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename
from docx import Document
import PyPDF2

app = Flask(__name__)
app.secret_key = "resume-analyzer-dev"

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

SKILL_KEYWORDS = [
    "python",
    "java",
    "javascript",
    "typescript",
    "sql",
    "aws",
    "docker",
    "kubernetes",
    "git",
    "react",
    "node",
    "flask",
    "django",
    "fastapi",
    "machine learning",
    "ai",
    "data analysis",
    "tableau",
    "excel",
    "pandas",
    "numpy",
    "spark",
    "azure",
    "linux",
    "api",
    "rest",
    "graphql",
    "microservices",
    "devops",
    "ci/cd",
    "testing",
]

ROLE_HINTS = {
    "python": ["python", "py", "django", "flask", "fastapi", "pandas", "numpy"],
    "aws": ["aws", "amazon web services", "ec2", "s3", "lambda"],
    "docker": ["docker", "containerization", "containers"],
    "sql": ["sql", "postgres", "postgresql", "mysql", "database"],
    "javascript": ["javascript", "js", "node", "react", "typescript"],
    "java": ["java", "spring", "hibernate"],
    "devops": ["devops", "ci/cd", "jenkins", "terraform", "kubernetes"],
    "data analysis": ["data analysis", "analytics", "tableau", "excel", "power bi"],
    "machine learning": ["machine learning", "ml", "ai", "pytorch", "tensorflow"],
}

STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "have",
    "that",
    "this",
    "your",
    "were",
    "will",
    "can",
    "into",
    "been",
    "work",
    "resume",
    "experience",
    "skills",
}

AI_PATTERNS = [
    r"\bchatgpt\b",
    r"\bopenai\b",
    r"\bai-generated\b",
    r"\bgenerative ai\b",
    r"\bcrafted\b",
    r"\btailored\b",
    r"\bleveraging\b",
    r"\boptimized for\b",
]

GENERIC_PHRASES = [
    "dynamic",
    "results-driven",
    "collaborative",
    "passionate",
    "innovative",
    "strategic thinker",
    "strong communicator",
    "team player",
    "detail-oriented",
]


def extract_text(uploaded_file):
    filename = secure_filename(uploaded_file.filename)
    suffix = Path(filename).suffix.lower()
    saved_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    uploaded_file.save(saved_path)

    if suffix == ".txt":
        return saved_path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".docx":
        doc = Document(saved_path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text)

    if suffix == ".pdf":
        reader = PyPDF2.PdfReader(str(saved_path))
        text_parts = []
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)

    return saved_path.read_text(encoding="utf-8", errors="ignore")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def extract_skills(text: str):
    lowered = normalize_text(text)
    found = []
    for skill in SKILL_KEYWORDS:
        pattern = rf"\b{re.escape(skill)}\b"
        if re.search(pattern, lowered):
            found.append(skill)
    return found


def extract_role_skills(job_description: str | None):
    if not job_description:
        return []
    lowered = normalize_text(job_description)
    found = []
    for skill in SKILL_KEYWORDS:
        if re.search(rf"\b{re.escape(skill)}\b", lowered):
            found.append(skill)
    return found


def match_skills(resume_text: str, job_description: str | None):
    resume_skills = extract_skills(resume_text)
    target_skills = extract_role_skills(job_description)
    matched = []
    missing = []

    for skill in target_skills:
        if skill in resume_skills:
            matched.append(skill)
        else:
            hints = ROLE_HINTS.get(skill, [])
            if any(hint in normalize_text(resume_text) for hint in hints):
                matched.append(skill)
            else:
                missing.append(skill)

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "resume_skills": resume_skills,
        "target_skills": target_skills,
    }


def get_section_presence(text: str):
    lowered = normalize_text(text)
    return {
        "summary": bool(re.search(r"\b(summary|profile|about me|objective)\b", lowered)),
        "experience": bool(re.search(r"\b(experience|employment|work history|professional experience)\b", lowered)),
        "education": bool(re.search(r"\b(education|university|college|degree|school)\b", lowered)),
        "skills": bool(re.search(r"\b(skills|technologies|tools|competencies)\b", lowered)),
        "projects": bool(re.search(r"\b(projects|portfolio|achievements)\b", lowered)),
    }


def extract_keywords_from_job_description(text: str):
    lowered = normalize_text(text)
    words = re.findall(r"[a-zA-Z][a-zA-Z+.#/-]{2,}", lowered)
    return [word for word in words if word not in STOP_WORDS and len(word) > 2]


def detect_ai_generation(text: str):
    lowered = normalize_text(text)
    ai_hits = [pattern for pattern in AI_PATTERNS if re.search(pattern, lowered)]
    generic_hits = sum(1 for phrase in GENERIC_PHRASES if phrase in lowered)

    if any(token in lowered for token in ["chatgpt", "openai", "ai-generated", "generative ai"]):
        return {
            "label": "Likely AI-assisted",
            "confidence": "High",
            "reason": "The text references AI tools or AI-generated language.",
        }

    if ai_hits or generic_hits >= 3:
        return {
            "label": "Possibly AI-assisted",
            "confidence": "Medium",
            "reason": "The wording feels generic or templated, which can suggest AI assistance.",
        }

    return {
        "label": "Human-style",
        "confidence": "Low",
        "reason": "No obvious AI-generation signals were detected.",
    }


def build_strengths_and_gaps(skills_found, sections, quantified, impact_hits):
    strengths = []
    if sections["summary"]:
        strengths.append("Clear professional summary")
    if sections["experience"]:
        strengths.append("Experience section present")
    if skills_found:
        strengths.append("Relevant technical skills detected")
    if quantified:
        strengths.append("Quantified achievements included")
    if impact_hits:
        strengths.append("Impact-oriented wording detected")

    if not strengths:
        strengths.append("A solid base is present")

    gaps = []
    if not sections["summary"]:
        gaps.append("Add a strong summary at the top")
    if not sections["experience"]:
        gaps.append("Expand the work history section")
    if not sections["skills"]:
        gaps.append("Add a dedicated skills section")
    if not quantified:
        gaps.append("Include measurable outcomes and metrics")
    if not sections["projects"]:
        gaps.append("Highlight projects or achievements")
    if not skills_found:
        gaps.append("Strengthen technical skill coverage")

    if not gaps:
        gaps.append("No major gaps detected")

    return strengths[:4], gaps[:4]


def analyze_resume(text: str, job_description: str | None = None):
    normalized = normalize_text(text)
    skills_found = extract_skills(text)
    roles = match_skills(text, job_description)
    sections = get_section_presence(text)
    quantified = bool(re.search(r"\b\d+(%|\+| years?| months?|k|m)?\b", normalized))
    impact_words = [
        "improved",
        "increased",
        "reduced",
        "built",
        "led",
        "developed",
        "optimized",
        "launched",
        "delivered",
        "achieved",
    ]
    impact_hits = [word for word in impact_words if word in normalized]

    role_fit_score = 0
    if roles["target_skills"]:
        role_fit_score = round((len(roles["matched_skills"]) / len(roles["target_skills"])) * 100)
    else:
        role_fit_score = 20

    score = 45
    score += min(len(skills_found) * 6, 24)
    score += round(role_fit_score * 0.25)
    score += 8 if sections["summary"] else 0
    score += 8 if sections["experience"] else 0
    score += 6 if sections["education"] else 0
    score += 6 if sections["skills"] else 0
    score += 5 if quantified else 0
    score += 4 if impact_hits else 0
    score = min(score, 100)

    feedback = []
    if not sections["summary"]:
        feedback.append("Add a concise professional summary near the top of the resume.")
    if len(skills_found) < 3:
        feedback.append("Strengthen the skills section with role-relevant technologies and tools.")
    if not quantified:
        feedback.append("Include measurable outcomes such as percentages, revenue, or team size.")
    if not sections["projects"]:
        feedback.append("Showcase projects or notable achievements to demonstrate practical impact.")
    if roles["missing_skills"]:
        feedback.append(f"Add more evidence of the following role requirements: {', '.join(roles['missing_skills'])}.")
    if not feedback:
        feedback.append("The resume is already strong; keep the formatting clear and tailored to the target role.")

    job_keywords = []
    if job_description:
        job_keywords = [word for word in extract_keywords_from_job_description(job_description) if word in normalized]

    if roles["target_skills"]:
        summary = (
            f"Your resume looks {score}% aligned for the target role. "
            f"It highlights {len(skills_found)} relevant skill area(s) and shows {'good' if sections['experience'] else 'limited'} evidence of work history."
            f" The role-fit analysis found {len(roles['matched_skills'])} matching skill area(s) and {len(roles['missing_skills'])} gap(s) to strengthen."
        )
    else:
        summary = (
            f"Your resume looks {score}% aligned for the target role, but the role description did not contain enough clear skill signals for a strong fit comparison. "
            f"The analysis is based on structure and resume content rather than a detailed role-match check."
        )

    strengths, gaps = build_strengths_and_gaps(skills_found, sections, quantified, impact_hits)
    ai_signal = detect_ai_generation(text)
    breakdown = {
        "Role fit": role_fit_score,
        "Skill coverage": round(min(len(skills_found) * 6, 24) / 24 * 100),
        "Summary strength": 100 if sections["summary"] else 0,
        "Experience signal": 100 if sections["experience"] else 0,
        "Education coverage": 100 if sections["education"] else 0,
        "Section structure": 100 if sections["skills"] else 0,
        "Quantified impact": 100 if quantified else 0,
        "Impact language": 100 if impact_hits else 0,
    }

    return {
        "score": score,
        "skills_found": skills_found,
        "matched_skills": roles["matched_skills"],
        "missing_skills": roles["missing_skills"],
        "sections": sections,
        "quantified": quantified,
        "impact_hits": impact_hits,
        "feedback": feedback,
        "summary": summary,
        "job_keywords": job_keywords,
        "strengths": strengths,
        "gaps": gaps,
        "ai_signal": ai_signal,
        "breakdown": breakdown,
        "role_fit": role_fit_score,
    }


def process_uploaded_resume(uploaded_file, job_description: str | None = None):
    text = extract_text(uploaded_file)
    result = analyze_resume(text, job_description)
    result["filename"] = uploaded_file.filename
    session["last_result"] = result
    return result


def build_html_report(result: dict) -> str:
    skills = ", ".join(result.get("skills_found", [])) or "No skills detected"
    sections = ", ".join(
        f"{name}: {'present' if present else 'missing'}" for name, present in result.get("sections", {}).items()
    )
    feedback = "<br>".join(result.get("feedback", []))
    return f"""<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <title>Resume Report - {result.get('filename', 'resume')}</title>
    <style>body{{font-family:Arial,sans-serif;line-height:1.6;padding:24px;}} h1{{color:#2563eb;}} .card{{border:1px solid #dce7f3;border-radius:12px;padding:16px;margin-bottom:14px;}}</style>
  </head>
  <body>
    <h1>ResumePilot Report</h1>
    <p><strong>Filename:</strong> {result.get('filename', 'resume')}</p>
    <p><strong>Score:</strong> {result.get('score', 0)}/100</p>
    <div class=\"card\"><h2>Summary</h2><p>{result.get('summary', '')}</p></div>
    <div class=\"card\"><h2>Detected Skills</h2><p>{skills}</p></div>
    <div class=\"card\"><h2>Section Coverage</h2><p>{sections}</p></div>
    <div class=\"card\"><h2>Feedback</h2><p>{feedback}</p></div>
    <div class=\"card\"><h2>AI Signal</h2><p>{result.get('ai_signal', {}).get('label', 'Unknown')}</p></div>
  </body>
</html>"""


def build_pdf_report(result: dict) -> bytes:
    lines = [
        "ResumePilot Report",
        f"Filename: {result.get('filename', 'resume')}",
        f"Score: {result.get('score', 0)}/100",
        "",
        "Summary:",
        result.get("summary", ""),
        "",
        "Detected skills:",
        ", ".join(result.get("skills_found", [])) or "No skills detected",
        "",
        "Feedback:",
        *result.get("feedback", []),
    ]
    text = "\n".join(lines)
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream_text = "BT\n/F1 12 Tf\n72 760 Td\n"
    stream_text += "(" + escaped.replace("\n", ") Tj\n0 -14 Td\n(") + ") Tj\nET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length 0 >>\nstream\n\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    stream_obj = f"<< /Length {len(stream_text.encode('latin-1'))} >>\nstream\n{stream_text}\nendstream".encode("latin-1")
    objects[3] = stream_obj
    pdf_parts = [b"%PDF-1.4\n"]
    offsets = [0]
    for obj in objects:
        offsets.append(len(b"".join(pdf_parts)))
        pdf_parts.append(f"{len(objects.index(obj) + 1)} 0 obj\n".encode("latin-1"))
        pdf_parts.append(obj + b"\n")
    xref_offset = len(b"".join(pdf_parts))
    pdf_parts.append(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf_parts.append(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf_parts.append(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf_parts.append(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin-1"))
    return b"".join(pdf_parts)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        uploaded_file = request.files.get("resume")
        job_description = request.form.get("job_description", "").strip()

        if not uploaded_file or uploaded_file.filename == "":
            flash("Please upload a resume file.")
            return redirect(url_for("index"))

        result = process_uploaded_resume(uploaded_file, job_description)

    return render_template("index.html", result=result)


@app.route("/analyze", methods=["POST"])
def analyze():
    uploaded_file = request.files.get("resume")
    job_description = request.form.get("job_description", "").strip()

    if not uploaded_file or uploaded_file.filename == "":
        return jsonify({"success": False, "message": "Please upload a resume file."}), 400

    result = process_uploaded_resume(uploaded_file, job_description)
    html = render_template("results_fragment.html", result=result)
    return jsonify({"success": True, "result": result, "html": html})


@app.route("/report")
def report():
    result = session.get("last_result")
    filename = request.args.get("filename", result.get("filename", "resume-report") if result else "resume-report")
    fmt = request.args.get("format", "html").lower()

    if not result:
        flash("No analysis available yet.")
        return redirect(url_for("index"))

    safe_name = secure_filename(Path(filename).stem or "resume-report")
    if fmt == "pdf":
        response = make_response(build_pdf_report(result))
        response.headers["Content-Type"] = "application/pdf"
        response.headers["Content-Disposition"] = f"attachment; filename={safe_name}.pdf"
        return response

    response = make_response(build_html_report(result))
    response.headers["Content-Type"] = "text/html; charset=utf-8"
    response.headers["Content-Disposition"] = f"attachment; filename={safe_name}.html"
    return response


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
