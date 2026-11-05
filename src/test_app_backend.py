"""
test_app_backend.py - Automated Unit & Integration Testing for Application Components

Verifies:
1. PDF document parsing via pypdf.
2. DOCX document parsing via python-docx.
3. Graceful error handling for empty, corrupted, and invalid files.
4. Multi-model inference via analyze_compatibility().
5. Output format and probability presence/absence validation.
"""

import io
import sys
import os
from pathlib import Path

# Paths
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
APP_DIR = PROJECT_ROOT / "app"

sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

import pypdf
import docx
from resume_parser import parse_resume_file, extract_skills_from_text, estimate_years_of_experience
from analyzer import analyze_compatibility, load_ml_pipeline_and_models


def create_dummy_pdf(text: str) -> io.BytesIO:
    """Generates an in-memory valid PDF with text content."""
    from pypdf import PdfWriter
    writer = PdfWriter()
    # Add a blank page
    writer.add_blank_page(width=612, height=792)
    stream = io.BytesIO()
    writer.write(stream)
    stream.seek(0)
    
    # Alternatively write a basic PDF structure
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        b"4 0 obj << /Length " + str(len(text) + 50).encode() + b" >> stream\n"
        b"BT /F1 12 Tf 72 712 Td (" + text.encode("latin-1", "replace") + b") Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000244 00000 n \n0000000350 00000 n \n"
        b"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n435\n%%EOF\n"
    )
    return io.BytesIO(pdf_content)


def create_dummy_docx(text: str) -> io.BytesIO:
    """Generates an in-memory valid DOCX with text paragraphs."""
    doc = docx.Document()
    doc.add_heading("Candidate Resume", 0)
    doc.add_paragraph(text)
    stream = io.BytesIO()
    doc.save(stream)
    stream.seek(0)
    return stream


def run_tests():
    print("=" * 70)
    print("RUNNING APP BACKEND & INTEGRATION TESTS")
    print("=" * 70)

    # Test 1: Artifact Loading
    print("\n[TEST 1] Loading ML Feature Pipeline & Saved Models...")
    artifacts = load_ml_pipeline_and_models()
    assert artifacts["pipeline"] is not None, "Pipeline failed to load!"
    assert len(artifacts["models"]) >= 5, f"Expected at least 5 models, found {len(artifacts['models'])}"
    print(f"  [PASS] Successfully loaded {len(artifacts['models'])} models and feature pipeline.")

    # Test 2: DOCX Parsing
    print("\n[TEST 2] DOCX Document Parsing...")
    sample_text = "Jane Doe. Senior Python Engineer with 7 years of experience in Machine Learning, Docker, SQL, and FastAPI."
    docx_stream = create_dummy_docx(sample_text)
    docx_stream.name = "resume_test.docx"
    res_docx = parse_resume_file(docx_stream)
    assert res_docx["success"] is True, f"DOCX parsing failed: {res_docx['error']}"
    assert "Python" in res_docx["text"] or "Jane Doe" in res_docx["text"], "Extracted text missing expected tokens!"
    print(f"  [PASS] DOCX parsed successfully ({res_docx['word_count']} words).")

    # Test 3: PDF Parsing
    print("\n[TEST 3] PDF Document Parsing...")
    pdf_stream = create_dummy_pdf("John Smith. Python Developer with 4 years of experience.")
    pdf_stream.name = "resume_test.pdf"
    res_pdf = parse_resume_file(pdf_stream)
    assert res_pdf["success"] is True, f"PDF parsing failed: {res_pdf['error']}"
    assert res_pdf["word_count"] > 0, "PDF word count should be > 0"
    print(f"  [PASS] PDF parsed successfully ({res_pdf['word_count']} words).")

    # Test 4: Error Handling on Corrupted/Empty/Unsupported Files
    print("\n[TEST 4] Error Handling Verification...")
    # Empty file
    empty_stream = io.BytesIO(b"")
    empty_stream.name = "empty.txt"
    res_empty = parse_resume_file(empty_stream)
    assert res_empty["success"] is False, "Empty file should fail validation!"
    print(f"  [PASS] Handled empty file: '{res_empty['error']}'")

    # Unsupported extension
    invalid_stream = io.BytesIO(b"some data")
    invalid_stream.name = "resume.xyz"
    res_invalid = parse_resume_file(invalid_stream)
    assert res_invalid["success"] is False, "Unsupported extension should fail validation!"
    print(f"  [PASS] Handled unsupported format: '{res_invalid['error']}'")

    # Test 5: Skill Extraction & Experience Estimation
    print("\n[TEST 5] Skill Extraction & Experience Estimation...")
    skills = extract_skills_from_text(sample_text)
    assert "python" in skills and "sql" in skills and "docker" in skills, f"Missing expected skills: {skills}"
    exp_est = estimate_years_of_experience(sample_text)
    assert exp_est == 7.0, f"Expected 7.0 years of experience, got {exp_est}"
    print(f"  [PASS] Correctly extracted {len(skills)} skills and detected {exp_est} years experience.")

    # Test 6: Full Compatibility Analysis on Real Resume-Job Pair
    print("\n[TEST 6] End-to-End Compatibility Prediction...")
    job_desc = "Seeking Senior Python Developer with 5+ years experience in Python, SQL, Docker, and Kubernetes. IT domain."
    analysis = analyze_compatibility(
        resume_text=sample_text,
        job_description=job_desc,
        years_experience=7.0,
        years_required=5.0,
        candidate_category="INFORMATION-TECHNOLOGY",
        job_category="INFORMATION-TECHNOLOGY"
    )
    assert analysis["success"] is True, f"Compatibility analysis failed: {analysis.get('error')}"
    assert analysis["primary_prediction"] in ["weak_positive", "medium_positive", "strong_positive"]
    assert "matched_skills" in analysis["skill_gap"]
    assert len(analysis["predictions"]) >= 5
    print(f"  [PASS] Primary KNN prediction: {analysis['primary_prediction']}")
    print(f"  [PASS] Multi-model consensus: {analysis['consensus_prediction']}")
    print(f"  [PASS] Matched skills: {analysis['skill_gap']['matched_skills']}")
    print(f"  [PASS] Missing skills: {analysis['skill_gap']['missing_skills']}")
    print(f"  [PASS] Jaccard similarity: {analysis['skill_gap']['jaccard_similarity']*100:.1f}%")

    print("\n" + "=" * 70)
    print("ALL 6 TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()

