"""
Preprocessing utilities for Resume Job Compatibility Analyzer.
Includes text cleaning, missing value handling, label encoding,
and dataset merging functions for downstream ML pipelines.
"""

import re
import pandas as pd
import numpy as np
from typing import Tuple, Optional


def clean_text(text: Optional[str]) -> str:
    """
    Cleans raw text strings by:
    - Lowercasing
    - Stripping HTML/markup or excessive symbols
    - Normalizing whitespace
    
    Parameters:
        text (str or None): The raw text to clean.
        
    Returns:
        str: Cleaned, normalized string.
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # Lowercase
    cleaned = text.lower()
    # Replace newlines, tabs, and multiple spaces with a single space
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)
    # Remove non-alphanumeric characters except basic punctuation
    cleaned = re.sub(r"[^\w\s\.,;:/\-]", " ", cleaned)
    # Condense multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def handle_missing_values(df: pd.DataFrame, strategy: str = "auto") -> pd.DataFrame:
    """
    Handles missing values in tabular datasets.
    
    Numeric columns: Filled with median.
    Categorical / Object columns: Filled with 'Unknown' or empty string.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
        strategy (str): Imputation strategy ('auto').
        
    Returns:
        pd.DataFrame: Imputed copy of the dataframe.
    """
    df_clean = df.copy()
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            
    text_cols = df_clean.select_dtypes(include=["object"]).columns
    for col in text_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna("Unknown")
            
    return df_clean


def merge_match_data(
    matches_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    job_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges match scores with corresponding candidate and job metadata.
    
    Joins:
        matches_df + cv_df on candidate_id
        + job_df on job_id
        
    Returns:
        pd.DataFrame: Unified dataframe containing paired match rows with attributes.
    """
    # Merge matches with candidate CV info
    merged = matches_df.merge(
        cv_df,
        on="candidate_id",
        how="inner",
        suffixes=("", "_cv")
    )
    
    # Merge result with job posting info
    merged = merged.merge(
        job_df,
        on="job_id",
        how="inner",
        suffixes=("", "_job")
    )
    
    return merged

