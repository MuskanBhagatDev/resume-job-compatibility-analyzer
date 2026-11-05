# Resume Job Compatibility Analyzer

An end-to-end Machine Learning application that evaluates semantic and structured compatibility between candidate resumes and job postings using Natural Language Processing, feature engineering, and classical machine learning algorithms.

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://resume-job-compatibility-analyzer-muskandev.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-245B4A?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)

- **Live Application:** [resume-job-compatibility-analyzer-muskandev.streamlit.app](https://resume-job-compatibility-analyzer-muskandev.streamlit.app/)
- **GitHub Repository:** [github.com/MuskanBhagatDev/resume-job-compatibility-analyzer](https://github.com/MuskanBhagatDev/resume-job-compatibility-analyzer)

---

## 1. Overview

Matching job applicants with suitable job openings is a critical challenge in modern recruitment. Traditional keyword-search applicant tracking systems (ATS) often miss contextual nuances, while large language models (LLMs) can be computationally expensive, non-deterministic, and prone to hallucinations.

The **Resume Job Compatibility Analyzer** addresses this challenge with an interpretable, reproducible, and CPU-efficient machine learning solution. It evaluates resume-job alignment by extracting technical skills, computing semantic and structured similarities, and predicting compatibility tiers using trained classical machine learning classifiers—all without external API dependencies or GPU acceleration.

### Core Objectives:
- **Classical ML Efficiency:** Built entirely on classical machine learning algorithms optimized for low-latency inference on standard CPU hardware.
- **Methodological Integrity:** Enforces candidate-grouped data partitioning to eliminate identity leakage and quarantines synthetic sub-scores to prevent data contamination.
- **Interpretability & Transparency:** Provides clear skill gap diagnoses, set-theoretic overlap metrics, and multi-model consensus predictions rather than opaque black-box scores.

---

## 2. Key Features

- **Multi-Format Document Parsing:** Ingests candidate resumes from selectable PDF files (`pypdf`), Microsoft Word `.docx` documents (`python-docx`), or direct text paste with real-time word counting.
- **Transparent Skill Gap Diagnosis:** Matches candidate qualifications against job requirements using a curated technical taxonomy, categorizing competencies into:
  - **Matched Skills:** Requirements satisfied by the candidate.
  - **Missing Skills:** Requirements not present in the candidate resume.
  - **Additional Skills:** Candidate strengths beyond the explicit job requirements.
- **Quantitative Overlap Metrics:** Computes set-theoretic Jaccard similarity, job requirement coverage percentage, and experience gap (candidate years vs. required years).
- **Multi-Model Compatibility Classification:** Evaluates input pairs across five trained classifiers, featuring **K-Nearest Neighbors ($k=5$)** as the primary engine alongside consensus predictions from Linear SVM, Random Forest, Logistic Regression, and a Multi-Layer Perceptron (MLP).
- **Interactive Model Diagnostics:** Built-in analytics dashboard displaying normalized confusion matrices, scree plots, clustering projections, and benchmark metric comparisons directly from verified test runs.

---

## 3. How It Works

```
Candidate Resume (PDF / DOCX / TXT)
                 │
                 ▼
     [ Document Text Extraction ]
                 │
                 ├──────────────────────────────┐
                 ▼                              ▼
    [ Skill Taxonomy Extraction ]    [ Sublinear TF-IDF Vectorizer ]
                 │                              │
                 ▼                              ▼
    [ Structured Feature Vector ]    [ 1,500 Unigram/Bigram Dims ]
    - TF-IDF Cosine Similarity                  │
    - Jaccard Skill Overlap                     │
    - Experience Difference                     │
    - Domain Match Indicator                    │
    - Word Count Ratios                         │
                 │                              │
                 └──────────────┬───────────────┘
                                ▼
               [ Compatibility Feature Pipeline ]
                                │
                                ▼
            [ Classical ML Multi-Model Inference ]
              - K-Nearest Neighbors (Primary)
              - Support Vector Machine (Linear SVM)
              - Random Forest Classifier
              - Logistic Regression (Softmax)
              - Multi-Layer Perceptron (MLP)
                                │
                                ▼
        [ Decision Dashboard & Skill Gap Report ]
        - Primary Compatibility Tier
        - Multi-Model Agreement Consensus
        - Calibrated Class Probabilities
        - Matched & Missing Skills Breakdown
```

1. **Document Ingestion:** The user uploads a resume (PDF/DOCX) or pastes text alongside job description requirements. Text layers and experience values are extracted.
2. **Feature Transformation:** The input pair is processed through a pre-fitted feature pipeline that generates a 1,500-dimensional TF-IDF text matrix and 16 standardized structured features (experience difference, cosine similarity, Jaccard overlap, category match, text lengths).
3. **Inference Execution:** Pre-trained model weights are evaluated without re-fitting to ensure 100% distribution consistency with training data.
4. **Diagnostic Output:** The system outputs the predicted compatibility class (`Weak Positive`, `Medium Positive`, or `Strong Positive`), model agreement consensus, and a visual skill gap diagnosis.

---

## 4. Machine Learning Methodology

### Dataset & Preprocessing
The model is trained and evaluated on the Hugging Face benchmark dataset [`nonameee12233/job-resume-matching`](https://huggingface.co/datasets/nonameee12233/job-resume-matching), containing:
- **cv.csv:** 9,544 candidate resume profiles across 5 industry sectors.
- **job.csv:** 1,167 job postings with detailed descriptions and requirements.
- **matches.csv:** 122,560 pairwise compatibility records.
- **Merged Dataset:** 122,560 match pairs representing 9,532 unique candidates and 1,164 unique jobs.

### Data Leakage Safeguards
A rigorous leakage audit identified that synthetic sub-scores (`semantic_score`, `skill_score`, `position_score`, `category_score`, `experience_score`, `education_score`, `candidate_rank`) present in the raw match table had mathematical dependencies on the target score. To prevent data contamination:
- All synthetic scores and continuous targets were quarantined and excluded from model inputs.
- Classifiers rely exclusively on legitimate resume text, job description text, and verified structured attributes.

### Candidate-Grouped Train / Validation / Test Splitting
To evaluate genuine real-world generalization and prevent candidate identity memorization, the dataset was partitioned using candidate-grouped splitting:

| Partition | Record Count | Percentage | Candidate Overlap | Purpose |
| :--- | :---: | :---: | :---: | :--- |
| **Training Set** | 85,392 | 69.67% | Zero (Disjoint) | Pipeline fitting and model weight optimization |
| **Validation Set** | 18,484 | 15.08% | Zero (Disjoint) | Hyperparameter tuning and model selection |
| **Held-Out Test Set** | 18,684 | 15.24% | Zero (Disjoint) | Final single-pass unbiased evaluation |
| **Total Merged** | **122,560** | **100.00%** | **0 Overlap** | **Grouped by `candidate_id` (~70:15:15)** |

Because candidates are strictly disjoint across partitions, models cannot memorize individual profiles and must learn transferable compatibility patterns.

### Feature Engineering
- **Text Features (1,500 dimensions):** Unigram and bigram TF-IDF vectorization with sublinear term-frequency scaling, English stop-word filtering, and vocabulary learned strictly on training data.
- **Structured Features (16 dimensions):** Standardized numerical features including experience gap ($\text{years\_experience} - \text{years\_required}$), domain match indicator, text length ratios, skill counts, Jaccard skill similarity, and TF-IDF cosine similarity.

---

## 5. Model Performance

All models were evaluated using candidate-grouped validation and test sets. Models were ranked strictly by **Validation Macro F1** to prevent test-set tuning bias:

| Rank | Model | Feature Representation | Validation Accuracy | Validation Macro F1 | Test Accuracy | Test Macro F1 | Test F1 (Strong) | Fit Time (s) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **K-Nearest Neighbors (KNN)** | Structured (16 dims) | **82.04%** | **0.7269** | **81.40%** | **0.7421** | **0.6111** | 0.12s |
| **2** | **Support Vector Machine (Linear SVM)** | All Features (1,516 dims) | 82.18% | 0.7082 | 81.80% | 0.6913 | 0.4404 | 66.35s |
| **3** | **Random Forest** | Structured (16 dims) | 80.76% | 0.6759 | 80.56% | 0.6825 | 0.4408 | 2.48s |
| **4** | **Logistic Regression (Multinomial)** | All Features (1,516 dims) | 79.23% | 0.6395 | 79.28% | 0.6313 | 0.2914 | 58.77s |
| **5** | **Neural Network (MLP)** | Structured (16 dims) | 78.30% | 0.6313 | 77.73% | 0.6157 | 0.2786 | 48.79s |
| **6** | **Naive Bayes (MultinomialNB)** | TF-IDF Text (1,500 dims) | 64.14% | 0.4548 | 62.73% | 0.4405 | 0.0671 | 0.27s |
| **7** | **Decision Tree (Entropy)** | Structured (16 dims) | 43.36% | 0.3363 | 41.67% | 0.3235 | 0.0419 | 0.51s |

### Why K-Nearest Neighbors (KNN) Was Selected as Primary Classifier:
1. **Top Macro F1 Score:** KNN achieved the highest validation Macro F1 score (**0.7269**) and test Macro F1 (**0.7421**), demonstrating superior balance across all three classes.
2. **Minority Class Sensitivity:** It achieved an F1 score of **0.6111** on the critical `Strong Positive` minority class, outperforming Linear SVM (0.4404) and Random Forest (0.4408).
3. **Local Geometric Discrimination:** The non-parametric instance-based neighborhood search ($k=5$, Euclidean distance on normalized structured features) captures localized compatibility clusters more effectively than global linear hyperplanes.
4. **Computational Efficiency:** Minimal training time (0.12s) and low memory usage make it well-suited for interactive web deployment.

---

## 6. Technology Stack

- **Core Language:** Python 3.10+
- **Machine Learning & Analytics:** Scikit-Learn, NumPy, Pandas, SciPy, Joblib
- **Web Application & UI:** Streamlit
- **Document Processing:** PyPDF (`pypdf`), Python-Docx (`python-docx`)
- **Data Visualization:** Matplotlib

---

## 7. Project Structure

```text
resume-job-compatibility-analyzer/
├── .streamlit/
│   └── config.toml               # Minimal SaaS light theme & server settings
├── app/
│   ├── streamlit_app.py          # Main application entry point & page router
│   ├── components.py             # UI components, KPI cards, and Lucide SVG icons
│   ├── resume_parser.py          # PDF, DOCX, and plain text extraction utilities
│   └── analyzer.py               # Core inference engine & skill gap diagnosis
├── data/
│   └── .gitkeep                  # Data directory (raw CSVs downloaded via script)
├── models/
│   ├── feature_pipeline.joblib   # Fitted TF-IDF and structured feature scaler
│   ├── k_nearest_neighbors.joblib# Primary classifier (Validation Macro F1: 0.7269)
│   ├── support_vector_machine.joblib
│   ├── random_forest.joblib
│   ├── logistic_regression.joblib
│   └── bpnn_mlp.joblib
├── outputs/
│   ├── plots/                    # High-resolution evaluation and diagnostic figures
│   └── results/                  # Quantitative benchmark comparison CSV tables
├── src/
│   ├── config.py                 # Dynamic path resolution and schema definitions
│   ├── data_loader.py            # Dataset loading and validation routines
│   ├── download_data.py          # Automated Hugging Face dataset downloader
│   ├── feature_engineering.py    # Feature extraction pipeline implementation
│   ├── leakage_audit.py          # Data leakage verification proofs
│   ├── ml_dataset.py             # Candidate-grouped reproducible split generator
│   ├── test_app_backend.py       # Automated backend integration test suite
│   └── test_app_pages.py         # Automated Streamlit page router test suite
├── .gitignore                    # Excludes raw CSVs and offline training caches
├── requirements.txt              # Production application dependencies
└── README.md                     # Project documentation
```

---

## 8. Local Installation and Execution

### Prerequisites:
- Python 3.10, 3.11, or 3.12
- Git
- Standard computer hardware (No GPU required)

### Step 1: Clone the Repository
```bash
git clone https://github.com/MuskanBhagatDev/resume-job-compatibility-analyzer.git
cd resume-job-compatibility-analyzer
```

### Step 2: Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run Verification Tests
```bash
# Verify model loading, document parsing, and inference pipeline
python src/test_app_backend.py

# Verify Streamlit page router execution
python src/test_app_pages.py
```

### Step 5: Launch the Application
```bash
streamlit run app/streamlit_app.py
```
*The web dashboard will automatically launch at `http://localhost:8501`.*

---

## 9. Live Cloud Deployment

The live application is deployed on **Streamlit Community Cloud**:

🔗 **[https://resume-job-compatibility-analyzer-muskandev.streamlit.app/](https://resume-job-compatibility-analyzer-muskandev.streamlit.app/)**

### Deployment Configuration:
- **Repository:** `MuskanBhagatDev/resume-job-compatibility-analyzer`
- **Branch:** `main`
- **Main file path:** `app/streamlit_app.py`
- **Runtime Environment:** Python 3.11 on standard cloud container
- **Storage Footprint:** ~40.6 MB total repository size; all required model weights are tracked directly in Git without requiring Git LFS.

---

## 10. Limitations and Academic Disclaimer

- **Algorithmic Compatibility Estimation:** Predictions reflect algorithmic patterns learned from the underlying benchmark dataset's scoring rubric. The application is designed for research, educational, and decision-support purposes; it does not replace human judgment or make autonomous hiring decisions.
- **Calibrated vs. Non-Probabilistic Outputs:** Probability values are presented only for classifiers with calibrated probabilistic formulations (KNN, Random Forest, Logistic Regression, MLP). Linear SVM outputs are reported as direct geometric hyperplane decisions without fabricated confidence scores.
- **Domain Coverage:** The models were trained across five industry domains (`Information Technology`, `Finance`, `Sales`, `Business Development`, and `Human Resources`). Inputs from unrepresented industries are evaluated using generalized semantic and structured overlap.

---

## 11. Author

**Created by Muskan**  
GitHub: [@MuskanBhagatDev](https://github.com/MuskanBhagatDev)  
Repository: [resume-job-compatibility-analyzer](https://github.com/MuskanBhagatDev/resume-job-compatibility-analyzer)
