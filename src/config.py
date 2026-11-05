"""
Configuration module for Resume Job Compatibility Analyzer.
Handles directory paths, dataset file names, column constants,
and global plotting settings.
"""

import os
from pathlib import Path

# Resolve base project directory dynamically
CURRENT_FILE = Path(__file__).resolve()
# If this file is in <project_root>/src/config.py, parent is src, parent.parent is project_root
PROJECT_ROOT = CURRENT_FILE.parent.parent

# Core Directories
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
RESULTS_DIR = OUTPUTS_DIR / "results"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
APP_DIR = PROJECT_ROOT / "app"

# Ensure runtime directories exist
for directory in [DATA_DIR, OUTPUTS_DIR, PLOTS_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Dataset file paths
CV_PATH = DATA_DIR / "cv.csv"
JOB_PATH = DATA_DIR / "job.csv"
MATCHES_PATH = DATA_DIR / "matches.csv"

# Expected Column Schemas
CV_COLUMNS = [
    "candidate_id",
    "target_position",
    "resume_text",
    "clean_skills",
    "education_text",
    "experience_text",
    "years_experience",
    "inferred_category"
]

JOB_COLUMNS = [
    "job_id",
    "job_title",
    "category",
    "job_description",
    "job_text",
    "clean_skills",
    "education_text",
    "years_required"
]

MATCH_SCORE_COLUMNS = [
    "semantic_score",
    "skill_score",
    "position_score",
    "category_score",
    "experience_score",
    "education_score"
]

TARGET_COLUMNS = [
    "final_score",
    "label_strength"
]

# Visualization Settings
PLOT_STYLE = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in os.environ else "default"
FIGURE_DPI = 300
PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#2ca02c"
ACCENT_COLOR = "#ff7f0e"
PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]

