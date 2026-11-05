"""
resume_parser.py - Robust Document Parsing & Skill Extraction Module

Handles ingestion of resumes in PDF, DOCX, and TXT formats using pypdf and python-docx.
Extracts clean plain text, validates document content, extracts candidate skills,
and provides graceful error handling for corrupted, empty, or encrypted files.
"""

import io
import re
from typing import Dict, List, Optional, Set, Tuple, Any

import pypdf
import docx


# Common technical & professional skill lexicon for keyword extraction
SKILL_LEXICON = {
    # Programming & Development
    "python", "java", "c++", "c#", "c", "javascript", "typescript", "php", "ruby", "golang", "go",
    "rust", "swift", "kotlin", "scala", "r", "perl", "bash", "shell", "powershell", "html", "css",
    "html5", "css3", "sass", "sql", "nosql", "graphql", "rest api", "rest", "soap", "json", "xml",
    # Frameworks & Libraries
    "react", "react.js", "angular", "vue", "vue.js", "node.js", "nodejs", "express", "django",
    "flask", "fastapi", "spring", "spring boot", "asp.net", ".net", "laravel", "rails",
    "pandas", "numpy", "scikit-learn", "scipy", "matplotlib", "seaborn", "tensorflow", "pytorch",
    "keras", "opencv", "nltk", "spacy", "huggingface", "llm", "transformers", "xgboost", "lightgbm",
    # Databases & Big Data
    "postgresql", "postgres", "mysql", "sqlite", "oracle", "sql server", "mongodb", "redis",
    "cassandra", "dynamodb", "elasticsearch", "neo4j", "hadoop", "spark", "hive", "kafka", "snowflake",
    # Cloud, DevOps & Tools
    "aws", "amazon web services", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
    "jenkins", "gitlab ci", "github actions", "terraform", "ansible", "ci/cd", "linux", "git",
    "jira", "confluence", "agile", "scrum", "kanban", "postman", "linux/unix",
    # Business, Finance & Analytics
    "excel", "advanced excel", "power bi", "tableau", "sap", "erp", "salesforce", "crm",
    "financial modeling", "accounting", "auditing", "budgeting", "forecasting", "taxation",
    "financial analysis", "risk management", "investment banking", "portfolio management",
    "market research", "lead generation", "b2b sales", "b2c sales", "cold calling",
    "business analysis", "data analysis", "project management", "pmp", "product management",
    # Human Resources & Management
    "recruitment", "talent acquisition", "onboarding", "employee engagement", "payroll",
    "hris", "performance management", "labor laws", "conflict resolution", "hr compliance",
    "leadership", "team management", "communication", "negotiation", "problem solving",
    "strategic planning", "customer service", "public relations"
}


def clean_extracted_text(text: str) -> str:
    """Normalizes whitespace and removes unprintable control characters."""
    if not text:
        return ""
    # Replace non-breaking spaces and excessive whitespace
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    # Remove null characters and non-printable bytes
    text = "".join(ch for ch in text if ch.isprintable() or ch in " \n")
    return text.strip()


def extract_text_from_pdf(file_input: Any) -> Tuple[str, Optional[str]]:
    """
    Extracts text from a PDF file using pypdf.
    
    Args:
        file_input: File-like object (BytesIO) or file path.
        
    Returns:
        Tuple of (extracted_text, error_message).
    """
    try:
        if isinstance(file_input, (str, bytes)):
            stream = io.BytesIO(file_input) if isinstance(file_input, bytes) else open(file_input, "rb")
        else:
            stream = file_input
            stream.seek(0)

        reader = pypdf.PdfReader(stream)
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                return "", "The PDF file is password protected or encrypted. Please upload an unlocked PDF."

        extracted_pages = []
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted_pages.append(page_text)

        full_text = clean_extracted_text(" ".join(extracted_pages))
        if not full_text:
            return "", "The PDF appears to be empty or contains scanned images without selectable text."
            
        return full_text, None

    except pypdf.errors.PdfReadError as e:
        return "", f"Corrupted or invalid PDF format: {str(e)}"
    except Exception as e:
        return "", f"Error reading PDF file: {str(e)}"


def extract_text_from_docx(file_input: Any) -> Tuple[str, Optional[str]]:
    """
    Extracts text from a DOCX file using python-docx.
    
    Args:
        file_input: File-like object (BytesIO) or file path.
        
    Returns:
        Tuple of (extracted_text, error_message).
    """
    try:
        if isinstance(file_input, (str, bytes)):
            stream = io.BytesIO(file_input) if isinstance(file_input, bytes) else open(file_input, "rb")
        else:
            stream = file_input
            stream.seek(0)

        doc = docx.Document(stream)
        text_parts = []
        
        # Extract from paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text.strip())
                
        # Extract from table cells (resumes often use tables for layout)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip() and cell.text.strip() not in text_parts:
                        text_parts.append(cell.text.strip())

        full_text = clean_extracted_text(" ".join(text_parts))
        if not full_text:
            return "", "The DOCX document contains no readable text paragraphs or tables."
            
        return full_text, None

    except Exception as e:
        return "", f"Error reading DOCX file: {str(e)}"


def parse_resume_file(uploaded_file: Any) -> Dict[str, Any]:
    """
    Unified entry point for parsing uploaded resume files from Streamlit.
    Supports PDF, DOCX, and TXT files.
    
    Returns a dictionary:
    {
        "success": bool,
        "text": str,
        "filename": str,
        "file_type": str,
        "word_count": int,
        "char_count": int,
        "error": Optional[str]
    }
    """
    if uploaded_file is None:
        return {
            "success": False,
            "text": "",
            "filename": "",
            "file_type": "",
            "word_count": 0,
            "char_count": 0,
            "error": "No file uploaded."
        }

    filename = getattr(uploaded_file, "name", "uploaded_document")
    ext = filename.split(".")[-1].lower() if "." in filename else ""

    text = ""
    error = None

    if ext == "pdf":
        text, error = extract_text_from_pdf(uploaded_file)
    elif ext in ("docx", "doc"):
        if ext == "doc":
            return {
                "success": False,
                "text": "",
                "filename": filename,
                "file_type": ext,
                "word_count": 0,
                "char_count": 0,
                "error": "Legacy .doc format is not supported directly. Please convert your file to .docx or .pdf."
            }
        text, error = extract_text_from_docx(uploaded_file)
    elif ext in ("txt", "text"):
        try:
            raw_bytes = uploaded_file.read()
            text = clean_extracted_text(raw_bytes.decode("utf-8", errors="ignore"))
            if not text:
                error = "The uploaded text file is empty."
        except Exception as e:
            error = f"Error reading text file: {str(e)}"
    else:
        error = f"Unsupported file extension '.{ext}'. Please upload a PDF, DOCX, or TXT document."

    word_count = len(text.split()) if text else 0
    char_count = len(text) if text else 0

    return {
        "success": error is None and word_count > 0,
        "text": text,
        "filename": filename,
        "file_type": ext,
        "word_count": word_count,
        "char_count": char_count,
        "error": error
    }


def extract_skills_from_text(text: str, custom_skills: Optional[List[str]] = None) -> List[str]:
    """
    Identifies relevant technical and professional skills from text
    by matching against the skill lexicon and any user-specified custom skills.
    
    Returns a sorted list of unique lower-case skill names.
    """
    if not text:
        return []

    text_lower = text.lower()
    # Normalize punctuation for token boundary checks
    padded_text = " " + re.sub(r"[^\w\s\+\#\./\-]", " ", text_lower) + " "

    found_skills: Set[str] = set()

    # Search known lexicon
    for skill in SKILL_LEXICON:
        # Exact word/phrase boundary matching
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, padded_text):
            found_skills.add(skill)

    # Check custom skills if supplied
    if custom_skills:
        for c_skill in custom_skills:
            clean_c = c_skill.strip().lower()
            if clean_c and clean_c in text_lower:
                found_skills.add(clean_c)

    return sorted(list(found_skills))


def estimate_years_of_experience(text: str) -> float:
    """
    Heuristically extracts years of experience mentioned in resume text.
    Falls back to a reasonable default (3.0 years) if none detected.
    """
    if not text:
        return 0.0

    patterns = [
        r"(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)",
        r"(?:experience|exp)\s*(?:of)?\s*:?\s*(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?)",
        r"(?:over|more than)\s+(\d+(?:\.\d+)?)\s*(?:years?|yrs?)",
    ]
    
    found_years = []
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        for m in matches:
            try:
                val = float(m)
                if 0.0 <= val <= 45.0:  # Sensible range check
                    found_years.append(val)
            except ValueError:
                continue

    if found_years:
        # Return maximum found explicit total experience
        return float(max(found_years))
        
    return 3.0  # Academic default median

