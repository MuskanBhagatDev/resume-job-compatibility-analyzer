"""
Feature Engineering Module for Resume Job Compatibility Analyzer.
Academic ML Lab Mini-Project - Phase 2 (Step 2).

Implements:
1. Text cleaning and normalization for resume and job description text
2. Train-only TF-IDF vectorization (unigrams + bigrams, stop words removed)
3. Resume-Job TF-IDF cosine similarity calculation
4. Skill overlap Jaccard similarity calculation
5. Engineered structured attributes (experience gap, category match, word counts, skill ratios)
6. Feature scaling and sparse/dense representations for diverse classical ML models
7. Pipeline persistence to models/feature_pipeline.joblib
"""

import sys
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Ensure path resolution
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import preprocessing


def compute_skill_jaccard(skills_cv: Any, skills_job: Any) -> float:
    """
    Computes Jaccard similarity between candidate skills and job requirements.
    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    """
    if pd.isna(skills_cv) or pd.isna(skills_job):
        return 0.0
    
    set_cv = set([s.strip().lower() for s in str(skills_cv).split(",") if s.strip()])
    set_job = set([s.strip().lower() for s in str(skills_job).split(",") if s.strip()])
    
    if not set_cv or not set_job:
        return 0.0
        
    intersection = len(set_cv & set_job)
    union = len(set_cv | set_job)
    return round(float(intersection / union), 6) if union > 0 else 0.0


class CompatibilityFeaturePipeline:
    """
    End-to-end feature extraction and transformation pipeline.
    Fits vectorizers and scalers exclusively on training data to prevent leakage.
    """
    
    def __init__(self, max_tfidf_features: int = 2500):
        self.max_tfidf_features = max_tfidf_features
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.scaler: Optional[StandardScaler] = None
        self.structured_feature_names: List[str] = []
        self.is_fitted: bool = False
        
    def _extract_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Constructs unscaled structured and interaction features."""
        df_out = df.copy()
        
        # 1. Experience Gap
        df_out["experience_gap"] = df_out["years_experience"].astype(float) - df_out["years_required"].astype(float)
        
        # 2. Category Match
        cand_cat = df_out["inferred_category"].astype(str).str.strip().str.upper()
        job_cat = df_out["job_category"].astype(str).str.strip().str.upper()
        df_out["category_match"] = (cand_cat == job_cat).astype(float)
        
        # 3. Text Lengths
        df_out["resume_word_count"] = df_out["resume_text"].astype(str).apply(lambda x: len(x.split())).astype(float)
        df_out["job_desc_word_count"] = df_out["job_description"].astype(str).apply(lambda x: len(x.split())).astype(float)
        
        # 4. Skill Counts and Ratios
        def get_skill_count(val):
            if pd.isna(val) or not isinstance(val, str):
                return 0.0
            return float(len([s for s in val.split(",") if s.strip()]))
            
        df_out["skill_count_cv"] = df_out["clean_skills_cv"].apply(get_skill_count)
        df_out["skill_count_job"] = df_out["clean_skills_job"].apply(get_skill_count)
        df_out["skill_count_ratio"] = df_out.apply(
            lambda r: (r["skill_count_cv"] / max(r["skill_count_job"], 1.0)),
            axis=1
        ).astype(float)
        
        # 5. Skill Jaccard Similarity (Optimized via dictionary set lookup)
        cand_skill_sets = {
            cid: set([s.strip().lower() for s in str(sk).split(",") if s.strip()])
            for cid, sk in zip(df_out["candidate_id"], df_out["clean_skills_cv"])
        }
        job_skill_sets = {
            jid: set([s.strip().lower() for s in str(sk).split(",") if s.strip()])
            for jid, sk in zip(df_out["job_id"], df_out["clean_skills_job"])
        }
        
        c_ids = df_out["candidate_id"].values
        j_ids = df_out["job_id"].values
        jaccards = np.zeros(len(df_out), dtype=np.float32)
        for i in range(len(df_out)):
            s_c = cand_skill_sets.get(c_ids[i])
            s_j = job_skill_sets.get(j_ids[i])
            if s_c and s_j:
                union_len = len(s_c | s_j)
                if union_len > 0:
                    jaccards[i] = len(s_c & s_j) / union_len
                    
        df_out["skill_jaccard_similarity"] = jaccards
        return df_out


    def fit(self, train_df: pd.DataFrame):
        """
        Fits TF-IDF vectorizer and structured feature scaler on training data ONLY.
        """
        print("[FEATURE PIPELINE] Fitting TF-IDF Vectorizer on training text corpus...")
        # Clean text
        train_resumes = train_df[["candidate_id", "resume_text"]].drop_duplicates()
        train_jobs = train_df[["job_id", "job_description"]].drop_duplicates()
        
        # Combined corpus for fitting vocabulary
        corpus = (
            train_resumes["resume_text"].fillna("").tolist() +
            train_jobs["job_description"].fillna("").tolist()
        )
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )
        self.vectorizer.fit(corpus)
        
        # Process structured attributes for training
        print("[FEATURE PIPELINE] Fitting StandardScaler on structured features...")
        df_feat = self._extract_base_features(train_df)
        
        # Compute TF-IDF Cosine Similarity for train_df
        df_feat["tfidf_similarity"] = self._compute_tfidf_similarity(df_feat)
        
        # One-hot encode category alignment
        categories = ["INFORMATION-TECHNOLOGY", "FINANCE", "SALES", "BUSINESS-DEVELOPMENT", "HR"]
        for cat in categories:
            col_name = f"is_{cat[:4].lower()}"
            df_feat[col_name] = (df_feat["job_category"].str.upper() == cat).astype(float)
            
        self.structured_feature_names = [
            "years_experience",
            "years_required",
            "experience_gap",
            "category_match",
            "resume_word_count",
            "job_desc_word_count",
            "skill_count_cv",
            "skill_count_job",
            "skill_count_ratio",
            "skill_jaccard_similarity",
            "tfidf_similarity"
        ] + [f"is_{cat[:4].lower()}" for cat in categories]
        
        self.scaler = StandardScaler()
        self.scaler.fit(df_feat[self.structured_feature_names].values)
        self.is_fitted = True
        print(f"[FEATURE PIPELINE] Fitted successfully. Structured features: {len(self.structured_feature_names)}, TF-IDF max features: {self.max_tfidf_features}")
        return self

    def _compute_tfidf_similarity(self, df: pd.DataFrame) -> np.ndarray:
        """
        Computes row-wise cosine similarity between resume and job TF-IDF vectors.
        Vectorizes unique entities once to ensure fast runtime.
        """
        if not self.vectorizer:
            raise ValueError("Vectorizer must be fitted before computing similarity.")
            
        # Get unique entities to transform fast
        unique_resumes = df[["candidate_id", "resume_text"]].drop_duplicates()
        unique_jobs = df[["job_id", "job_description"]].drop_duplicates()
        
        res_tfidf = self.vectorizer.transform(unique_resumes["resume_text"].fillna(""))
        job_tfidf = self.vectorizer.transform(unique_jobs["job_description"].fillna(""))
        
        # Map unique IDs to row index in the sparse matrix
        cand_to_idx = {cid: idx for idx, cid in enumerate(unique_resumes["candidate_id"])}
        job_to_idx = {jid: idx for idx, jid in enumerate(unique_jobs["job_id"])}
        
        # Vectorized array indexing
        c_indices = df["candidate_id"].map(cand_to_idx).fillna(0).astype(int).values
        j_indices = df["job_id"].map(job_to_idx).fillna(0).astype(int).values
        
        res_rows = res_tfidf[c_indices]
        job_rows = job_tfidf[j_indices]
        
        # Row-wise dot product is instantaneous in C/SciPy
        sims = np.asarray(res_rows.multiply(job_rows).sum(axis=1)).flatten()
        return np.clip(sims, 0.0, 1.0)


    def transform(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Transforms any dataset partition (train, val, test) into model-ready features.
        
        Returns a dictionary containing:
        - X_struct: Dense 2D numpy array of scaled structured features
        - X_tfidf: Sparse matrix of combined resume-job TF-IDF
        - X_all: Horizontally stacked sparse matrix [X_tfidf, X_struct]
        - y_clf: 1D array of label_strength classes
        - y_reg: 1D array of final_score values
        - feature_names_struct: List of structured column names
        """
        if not self.is_fitted:
            raise ValueError("FeaturePipeline must be fitted before transform.")
            
        df_feat = self._extract_base_features(df)
        df_feat["tfidf_similarity"] = self._compute_tfidf_similarity(df_feat)
        
        categories = ["INFORMATION-TECHNOLOGY", "FINANCE", "SALES", "BUSINESS-DEVELOPMENT", "HR"]
        for cat in categories:
            col_name = f"is_{cat[:4].lower()}"
            df_feat[col_name] = (df_feat["job_category"].str.upper() == cat).astype(float)
            
        # 1. Dense structured feature matrix (scaled)
        X_struct_raw = df_feat[self.structured_feature_names].values.astype(np.float32)
        X_struct_scaled = self.scaler.transform(X_struct_raw).astype(np.float32)
        
        # 2. Sparse TF-IDF representation (pair text)
        combined_text = (
            df_feat["resume_text"].fillna("") + " " +
            df_feat["clean_skills_cv"].fillna("") + " " +
            df_feat["clean_skills_job"].fillna("") + " " +
            df_feat["job_description"].fillna("")
        )
        X_tfidf = self.vectorizer.transform(combined_text)
        
        # 3. Unified combined representation
        X_struct_sparse = sp.csr_matrix(X_struct_scaled)
        X_all = sp.hstack([X_tfidf, X_struct_sparse], format="csr")
        
        # Targets
        y_clf = df["label_strength"].values if "label_strength" in df.columns else None
        y_reg = df["final_score"].values if "final_score" in df.columns else None
        
        return {
            "X_struct": X_struct_scaled,
            "X_tfidf": X_tfidf,
            "X_all": X_all,
            "y_clf": y_clf,
            "y_reg": y_reg,
            "feature_names_struct": self.structured_feature_names
        }

    def save(self, filepath: Optional[Path] = None):
        """Save fitted pipeline object to disk."""
        path = filepath or (config.MODELS_DIR / "feature_pipeline.joblib")
        joblib.dump(self, path)
        print(f"[PIPELINE SAVED] {path}")

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> "CompatibilityFeaturePipeline":
        """Load fitted pipeline object from disk."""
        path = filepath or (config.MODELS_DIR / "feature_pipeline.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pipeline model not found at {path}")
        return joblib.load(path)


def load_or_create_features(force_recompute: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Convenience function to load or generate transformed feature partitions.
    Uses candidate-grouped split (70/15/15, seed 42) and fitted pipeline.
    Caches transformed partitions to models/preprocessed_splits.joblib for instant loading.
    """
    cache_path = config.MODELS_DIR / "preprocessed_splits.joblib"
    if not force_recompute and os.path.exists(cache_path):
        print(f"[CACHE HIT] Loading preprocessed feature splits from {cache_path}...")
        splits = joblib.load(cache_path)
        return splits["train"], splits["val"], splits["test"]

    print("[COMPUTING] Building dataset and extracting feature splits...")
    import ml_dataset
    
    merged_raw, _ = ml_dataset.build_and_validate_merged_dataset()
    df_eng = ml_dataset.engineer_safe_features(merged_raw)
    train_df, val_df, test_df, _ = ml_dataset.split_dataset_by_candidate(df_eng, random_seed=42)
    
    pipe_path = config.MODELS_DIR / "feature_pipeline.joblib"
    if os.path.exists(pipe_path):
        pipe = CompatibilityFeaturePipeline.load(pipe_path)
    else:
        pipe = CompatibilityFeaturePipeline(max_tfidf_features=1500)
        pipe.fit(train_df)
        pipe.save()
        
    print("[TRANSFORMING] Transforming train, val, test partitions...")
    train_feats = pipe.transform(train_df)
    val_feats = pipe.transform(val_df)
    test_feats = pipe.transform(test_df)
    
    print(f"[CACHING] Saving preprocessed feature splits to {cache_path}...")
    joblib.dump({"train": train_feats, "val": val_feats, "test": test_feats}, cache_path, compress=3)
    print("[CACHING COMPLETE]")
    return train_feats, val_feats, test_feats

