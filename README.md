# Resume Job Compatibility Analyzer

**Academic Machine Learning Lab Mini-Project**  
**Complete Pipeline: EDA, Leakage Prevention, Feature Engineering, Classical ML Backend & Live Demonstration App**

---

## 1. Project Overview

The **Resume Job Compatibility Analyzer** is an academic Machine Learning laboratory mini-project designed to evaluate and predict compatibility between job applicant resumes and employer job descriptions using legitimate textual and structured features.

### Academic Core Principles:
- **Classical Machine Learning Focus:** Strictly avoids dependency on heavy deep learning architectures, Transformer embeddings, or external LLM APIs.
- **Hardware-Friendly:** Lightweight, CPU-optimized, and fully runnable on standard laptop hardware without GPU acceleration.
- **Strict Anti-Leakage Safeguards:** Rigorous leakage audit proves that all synthetic component scores (`semantic_score`, `skill_score`, etc.), raw ranking variables, and ground-truth targets are quarantined from model inputs.
- **Pedagogical Rigor:** Implements classical algorithms from the Machine Learning Laboratory syllabus with mathematical justifications, reproducible splits, publication-quality diagnostic plots, and an interactive Streamlit demonstration application.

---

## 2. Machine Learning Lab Algorithm Coverage

| Exp # | Algorithm / Technique | Implementation Module | Pedagogical Objective |
| :---: | :--- | :--- | :--- |
| **1** | **Feature Selection** | `src/feature_selection.py` | ANOVA $F$-test filter (`SelectKBest`, $k=500$) compared against full feature matrix. |
| **2** | **Linear & Ridge Regression** | `src/train_regressor.py` | Ordinary Least Squares (OLS) vs $L_2$ Regularized Ridge on continuous target `final_score`. |
| **3** | **Logistic Regression** | `src/train_classifiers.py` | Multi-class softmax with class balancing on all vs selected features. |
| **4** | **Support Vector Machine (SVM)** | `src/train_classifiers.py` | Maximum margin separation using Linear Support Vector Classifier with dual relaxation. |
| **5** | **Backpropagation Neural Net** | `src/train_bpnn.py` | Multi-Layer Perceptron (MLP) trained via mini-batch gradient descent and error backprop. |
| **6** | **PCA & TruncatedSVD** | `src/run_pca.py` | Dense covariance PCA vs Sparse Latent Semantic Analysis (LSA) on TF-IDF space. |
| **7** | **ID3 Decision Tree** | `src/train_id3.py` | Shannon Entropy ($H(S)$) and Information Gain ($IG$) partitioning with tree visualization. |
| **8** | **Random Forest** | `src/train_classifiers.py` | Ensemble bagging of 100 decorrelated decision trees with out-of-bag diversity. |
| **9** | **K-Nearest Neighbors (KNN)** | `src/train_classifiers.py` | Non-parametric instance-based spatial neighborhood search ($k=5$). |
| **10** | **Naive Bayes** | `src/train_classifiers.py` | Multinomial likelihood with Laplace smoothing on sparse text n-grams. |
| **11** | **K-Means Clustering** | `src/run_kmeans.py` | Unsupervised partitioning via Lloyd's algorithm with Elbow and Silhouette diagnostics. |

---

## 3. Project Directory Structure

```text
resume-job-analyzer/
│
├── .gitignore                      # Git exclusion rules for large files, raw data & caches
├── .streamlit/
│   └── config.toml                 # Minimal SaaS light theme and server configuration
│
├── app/                            # Interactive Live Demonstration Application
│   ├── streamlit_app.py            # Main entrypoint & multi-page router (Entry point for Streamlit)
│   ├── components.py               # Reusable minimal SaaS UI components, KPI cards & Lucide SVG icons
│   ├── resume_parser.py            # PDF (pypdf), DOCX (python-docx), and TXT text extractors
│   └── analyzer.py                 # Core inference, multi-model consensus & skill gap engine
│
├── data/                           # Data directory (Raw CSVs ignored by Git; download via script)
│   ├── .gitkeep                    # Preserves data directory structure in Git
│   ├── cv.csv                      # (185 MB) Raw candidate resumes (Hugging Face)
│   ├── job.csv                     # (31 MB) Raw job postings (Hugging Face)
│   └── matches.csv                 # (20 MB) Raw match records (Hugging Face)
│
├── notebooks/
│   └── 01_eda.ipynb                # Interactive Jupyter Notebook walkthrough
│
├── src/
│   ├── config.py                   # Centralized paths, schemas, and dynamic path resolution
│   ├── data_loader.py              # Ingestion, schema validation, and summary functions
│   ├── download_data.py            # Automated dataset downloader from Hugging Face
│   ├── preprocessing.py            # Text cleaning and preprocessing utilities
│   ├── leakage_audit.py            # Target relationship audit and leakage prevention proofs
│   ├── target_analysis.py          # Distribution analysis of classification and regression targets
│   ├── ml_dataset.py               # Merge validation and candidate-grouped reproducible splitting
│   ├── feature_engineering.py      # TF-IDF, Cosine Similarity, Jaccard Overlap, Structured Scaling
│   ├── feature_selection.py        # SelectKBest ANOVA F-test experiment
│   ├── train_classifiers.py        # Baselines: Logistic Regression, SVM, KNN, RF, Naive Bayes
│   ├── train_regressor.py          # Linear Regression and Ridge Regression with alpha analysis
│   ├── train_bpnn.py               # Experiment 5: Backpropagation Multi-Layer Perceptron (MLP)
│   ├── run_pca.py                  # Experiment 6: PCA and TruncatedSVD Dimensionality Reduction
│   ├── train_id3.py                # Experiment 7: Entropy / Information-Gain Decision Tree
│   ├── run_kmeans.py               # Experiment 11: K-Means Clustering & Contingency Analysis
│   ├── test_app_backend.py         # Automated integration and document parser test suite
│   ├── test_app_pages.py           # Automated Streamlit page router test suite
│   ├── eda_plots.py                # Matplotlib visualization routines
│   └── run_eda.py                  # Standalone CLI execution script for EDA
│
├── models/                         # Saved joblib model artifacts (Git tracked, ~32.7 MB total)
│   ├── feature_pipeline.joblib     # Precomputed fitted feature extraction pipeline (64 KB)
│   ├── k_nearest_neighbors.joblib  # Primary classifier (5.86 MB) - Val Macro F1: 0.7269
│   ├── support_vector_machine.joblib # LinearSVC classifier (37 KB)
│   ├── random_forest.joblib        # Random Forest classifier (25.57 MB)
│   ├── logistic_regression.joblib  # Softmax Logistic Regression (37 KB)
│   ├── bpnn_mlp.joblib             # Backpropagation Neural Network MLP (51 KB)
│   ├── naive_bayes.joblib          # Multinomial Naive Bayes (85 KB)
│   ├── kmeans_clusterer.joblib     # K-Means clustering model (342 KB)
│   ├── pca_svd_reducers.joblib     # TruncatedSVD dimensionality reducer (604 KB)
│   ├── id3_decision_tree.joblib    # Entropy decision tree (4 KB)
│   └── preprocessed_splits.joblib  # (374 MB - EXCLUDED from Git; offline training cache only)
│
├── outputs/
│   ├── plots/                      # 20 publication-grade diagnostic plots (300 DPI, ~7.6 MB)
│   │   ├── bpnn_loss_and_cm.png
│   │   ├── id3_decision_tree.png
│   │   ├── id3_feature_importance.png
│   │   ├── pca_svd_scree.png
│   │   ├── pca_struct_scatter.png
│   │   ├── kmeans_elbow_analysis.png
│   │   └── ... (All experimental plots)
│   └── results/                    # Quantitative benchmark metrics and evaluation tables (~90 KB)
│       ├── classification_model_comparison.csv
│       ├── regression_model_comparison.csv
│       ├── ml_dataset_summary.csv
│       └── ... (All summary metrics)
│
├── requirements.txt                # Production dependencies for local run and Streamlit Cloud
└── README.md                       # Comprehensive project documentation
```

---

## 4. Master Model Benchmark Comparison

All classification models evaluated on candidate-grouped splits (Train: 85,392 | Val: 18,484 | Test: 18,684).  
Ranked strictly by **Validation Macro F1** to prevent test-set data leakage:

| Rank | Model | Feature Representation | Fit Time (s) | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 | Test F1 (Strong) | Role in System |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **K-Nearest Neighbors** | Structured (16) | 0.12s | **82.04%** | **0.7269** | **81.40%** | **0.7421** | **0.6111** | **Primary Classifier** |
| **2** | **Support Vector Machine (Linear)** | All Features (1,516) | 66.35s | **82.18%** | **0.7082** | **81.80%** | **0.6913** | 0.4404 | Secondary Classifier |
| **3** | **Random Forest** | Structured (16) | 2.48s | 80.76% | 0.6759 | 80.56% | 0.6825 | 0.4408 | Ensemble Classifier |
| **4** | **Linear SVM (SelectKBest)** | Selected Features (500) | 27.18s | 77.79% | 0.6580 | 77.50% | 0.6401 | 0.3765 | Feature Selection Exp |
| **5** | **Logistic Regression** | All Features (1,516) | 58.77s | 79.23% | 0.6395 | 79.28% | 0.6313 | 0.2914 | Linear Softmax Baseline |
| **6** | **Backprop Neural Net (MLP)** | Structured (16) | 48.79s | 78.30% | 0.6313 | 77.73% | 0.6157 | 0.2786 | 3-Layer MLP Classifier |
| **7** | **Logistic Regression (SelectKBest)**| Selected Features (500) | 24.96s | 73.86% | 0.5773 | 73.46% | 0.5653 | 0.1976 | Feature Selection Exp |
| **8** | **Naive Bayes (Multinomial)** | TF-IDF Text (1,500) | 0.27s | 64.14% | 0.4548 | 62.73% | 0.4405 | 0.0671 | Probabilistic Baseline |
| **9** | **ID3 Decision Tree (Entropy, d=4)** | Structured (16) | 0.51s | 43.36% | 0.3363 | 41.67% | 0.3235 | 0.0419 | Interpretable Tree |

---

## 5. Live Demonstration Web Application

The live demonstration dashboard provides a clean graphical interface designed for academic lab presentation:

### Features:
1. **Multi-Format Document Parsing:**
   - Ingests PDF resumes via `pypdf` with selectable text extraction.
   - Ingests Microsoft Word `.docx` documents via `python-docx` (body text and tables).
   - Direct text area fallback with live word and character counters.
2. **Transparent Skill Gap Diagnosis:**
   - Matches candidate skills against job requirements using an internal technical taxonomy.
   - Clean color-coded chips for Matched, Missing, and Additional skills.
   - Set-theoretic Jaccard similarity and requirement coverage calculations.
3. **Multi-Model Inference & Consensus:**
   - Primary decision powered by **K-Nearest Neighbors ($k=5$)**.
   - Side-by-side consensus breakdown across Linear SVM, Random Forest, Logistic Regression, and MLP.
   - Strictly reports true model decisions (calibrated probabilities for probabilistic models; geometric hyperplane outputs for SVM).
4. **Interactive Model Diagnostics:**
   - Visualizes confusion matrices, scree plots, clustering projections, decision tree diagrams, and benchmark comparison tables.

---

## 6. Local Installation & Execution

### Prerequisites:
- Python 3.10, 3.11, 3.12, or 3.13
- Standard laptop hardware (No GPU required)

### Step 1: Clone or Open the Repository
```bash
git clone https://github.com/<your-username>/resume-job-analyzer.git
cd resume-job-analyzer
```

### Step 2: Set up a Virtual Environment
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS / Linux:
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run Automated Verification Tests
```bash
# Verify model loading, parsing, and inference pipeline
python src/test_app_backend.py

# Verify Streamlit page router execution
python src/test_app_pages.py
```

### Step 5: Launch the Streamlit Application
```bash
streamlit run app/streamlit_app.py
```
*The app will automatically open in your default browser at `http://localhost:8501`.*

---

## 7. Streamlit Community Cloud Deployment

The repository is structured to deploy directly to **Streamlit Community Cloud** with zero configuration friction.

### Why It Deploys Seamlessly:
1. **No External LLM / GPU Dependencies:** Runs entirely on standard cloud CPU instances.
2. **Repository Size < 45 MB:**
   - Raw datasets (`data/*.csv`, ~236 MB) and offline training caches (`models/preprocessed_splits.joblib`, ~374 MB) are excluded via `.gitignore`.
   - All trained model files required by the live app (`models/*.joblib`) total only **~32.7 MB**, with the largest single model (`random_forest.joblib`) at **25.57 MB** (well within GitHub's 100 MB limit).
   - Git LFS is **not required**.
3. **Dynamic Path Resolution:** Paths use `pathlib.Path` relative to the repository root. Zero hardcoded Windows drive letters or absolute paths exist in the codebase.
4. **Optimized Dependencies:** `requirements.txt` contains only runtime application packages, ensuring fast and reliable cloud builds.

### Step-by-Step Deployment Instructions:

#### Step 1: Push Code to GitHub
```bash
# Inside resume-job-analyzer directory
git init
git add .
git status
# Verify that data/*.csv and models/preprocessed_splits.joblib are NOT staged
git commit -m "Initial commit: Resume Job Compatibility Analyzer"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

#### Step 2: Deploy on Streamlit Community Cloud
1. Sign in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
2. Click **"New app"** (or **"Create app"**).
3. Select your repository: `<your-username>/<your-repo-name>`
4. Set the **Branch:** `main`
5. Set the **Main file path:** `app/streamlit_app.py`
6. Click **"Advanced settings..."**:
   - In the **Python version** dropdown, select `3.10` or `3.11` (or `3.12`).
   *(Note: Streamlit Community Cloud uses this dropdown rather than a `runtime.txt` file).*
7. Click **"Deploy!"**.
8. Within 1-2 minutes, your live web application will be accessible worldwide via your dedicated Streamlit URL.

---

## 8. Dataset Download (For Retraining Offline)

The raw dataset is not stored in Git due to size (~236 MB). If you wish to retrain the models from scratch locally:

```bash
# Downloads cv.csv, job.csv, and matches.csv from Hugging Face into data/
python src/download_data.py
```

Dataset Source:
- Hugging Face: [`nonameee12233/job-resume-matching`](https://huggingface.co/datasets/nonameee12233/job-resume-matching)

---

## 9. How to Re-Run Backend ML Lab Experiments

To execute and verify any backend experiment independently from the repository root:

```bash
# 1. Automated Integration & Parser Test Suite
python src/test_app_backend.py
python src/test_app_pages.py

# 2. EDA & Anti-Leakage Audit
python src/run_eda.py
python src/leakage_audit.py
python src/ml_dataset.py

# 3. Feature Selection & Baseline Classifiers
python src/feature_selection.py
python src/train_classifiers.py
python src/train_regressor.py

# 4. Neural Network, Dimensionality Reduction & Trees
python src/train_bpnn.py
python src/run_pca.py
python src/train_id3.py
python src/run_kmeans.py
```

---

## 10. Academic Notice & Methodological Disclaimer

- **Scoring Rubric Alignment:** Model predictions reflect algorithmic approximations of the underlying Hugging Face dataset's scoring patterns. The system serves strictly for academic ML lab evaluation and comparative algorithmic study; it does not replace real-world hiring decisions or human judgment.
- **Probabilistic vs Hyperplane Decisions:** Probability estimates are reported only for models with calibrated probabilistic formulations (e.g., KNN, Random Forest, Logistic Regression); non-probabilistic classifiers (such as Linear Support Vector Machines) report direct geometric hyperplane decisions.
