# -*- coding: utf-8 -*-
"""
paths.py
========
Centralized path management using pathlib for cross-platform compatibility.
"""

import os
from pathlib import Path

# Project root (electricity-theft-detection/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Core directories
DATASET_DIR = ROOT_DIR / "dataset"
MODEL_DIR   = ROOT_DIR / "model"
SRC_DIR     = ROOT_DIR / "src"
SCREENSHOTS = ROOT_DIR / "screenshots"

# Ensure directories exist
for d in [DATASET_DIR, MODEL_DIR, SCREENSHOTS]:
      d.mkdir(parents=True, exist_ok=True)

# Data files
FEAT_CSV    = DATASET_DIR / "features.csv"
PROC_CSV    = DATASET_DIR / "processed_hourly.csv"
RAW_ZIP     = DATASET_DIR / "LD2011_2014.txt.zip"
RAW_TXT     = DATASET_DIR / "LD2011_2014.txt"

# Model files
IF_PATH         = MODEL_DIR / "isolation_forest.pkl"
RF_PATH         = MODEL_DIR / "random_forest.pkl"
SCALER_PATH     = MODEL_DIR / "scaler.pkl"
THRESHOLD_PATH  = MODEL_DIR / "thresholds.json"
METRICS_PATH    = MODEL_DIR / "metrics.json"
FEAT_COLS_PATH  = MODEL_DIR / "feature_columns.pkl"

def get_relative_path(path: Path) -> str:
      """Return a string path relative to ROOT_DIR if possible, else absolute."""
      try:
                return str(path.relative_to(ROOT_DIR))
except ValueError:
        return str(path)
