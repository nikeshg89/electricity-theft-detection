# -*- coding: utf-8 -*-
"""
predict.py  (FULL REWRITE)
==========
Inference utilities for the Smart Electricity Theft Detection System.

Training unit = one 24-hour profile.
Features = same summary stats as computed in train.py.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

IF_PATH         = os.path.join(MODEL_DIR, "isolation_forest.pkl")
RF_PATH         = os.path.join(MODEL_DIR, "random_forest.pkl")
SCALER_PATH     = os.path.join(MODEL_DIR, "scaler.pkl")
THRESHOLD_PATH  = os.path.join(MODEL_DIR, "thresholds.json")

# Must match train.py FEATURE_COLS exactly
DEFAULT_FEATURE_ORDER = [
      "mean_cons",
      "std_cons",
      "max_cons",
      "min_cons",
      "max_min_diff",
      "night_sum",
      "day_sum",
      "night_vs_day_ratio",
      "peak_hour",
      "flatline_score",
      "spike_score",
]

LABEL_MAP = {0: "Normal Usage", 1: "Suspicious (Possible Theft)", 2: "Slightly Unusual"}


# == Model loader (cached in module globals) ==

_models  = {}
_scaler  = None
_thresh  = None


def load_models():
      global _models, _scaler, _thresh
      if _models:
                return _models

      if not os.path.exists(IF_PATH):
                raise FileNotFoundError("Models not found. Run training pipeline first.")

      _models["isolation_forest"] = joblib.load(IF_PATH)
      if os.path.exists(RF_PATH):
                _models["random_forest"] = joblib.load(RF_PATH)
                if os.path.exists(SCALER_PATH):
                      _scaler = joblib.load(SCALER_PATH)

    if os.path.exists(THRESHOLD_PATH):
              with open(THRESHOLD_PATH) as f:
                            _thresh = json.load(f)

          return _models



# == Feature extraction from 24-hour profile ==

def profile_to_features(profile: np.ndarray) -> np.ndarray:
      """
          Convert a 24-element hourly consumption profile into the 11-feature vector
              that matches train.py's feature extraction.
                  """
    hours = np.arange(24)
    mean_c = profile.mean()
    std_c  = profile.std()
    max_c  = profile.max()
    min_c  = profile.min()
    max_min = max_c - min_c

    night_mask = (hours < 6) | (hours >= 22)
    day_mask   = ~night_mask
    night_sum  = profile[night_mask].sum()
    day_sum    = profile[day_mask].sum()
    nd_ratio   = night_sum / (day_sum + 1e-6)

    peak_hour  = float(np.argmax(profile))
    flatline   = float((profile < 0.1).sum()) / 24.0
    spike      = float((profile > 2 * mean_c + 1e-6).sum()) / 24.0

    return np.array([
              mean_c, std_c, max_c, min_c, max_min,
              night_sum, day_sum, nd_ratio,
              peak_hour, flatline, spike
    ])


# == Prediction ==

def predict_single(hourly_readings: list, model: str = "isolation_forest") -> dict:
      """
          Predict a single 24-hour consumption profile.

              Parameters
                  ----------
                      hourly_readings : list of 24 floats (kWh per hour)
                          model           : 'isolation_forest' or 'random_forest'

                              Returns
                                  -------
                                      dict with keys: label, code, score, confidence
                                          """
    if len(hourly_readings) != 24:
              raise ValueError("Exactly 24 hourly readings required.")

    models = load_models()
    profile = np.array(hourly_readings, dtype=float).clip(0)

    # Extract features
    feat_vec = profile_to_features(profile).reshape(1, -1)

    # Apply scaler
    if _scaler is not None:
              feat_vec = _scaler.transform(feat_vec)


    # == Isolation Forest path ==
    if model == "isolation_forest":

              clf = models["isolation_forest"]
              score = float(clf.decision_function(feat_vec)[0])

        # Load dynamic thresholds
              if _thresh is not None:
                            normal_min     = _thresh.get("normal_min", -0.02)
                            suspicious_max = _thresh.get("suspicious_max", -0.10)
    else:
            # Safe defaults if thresholds file missing
                  normal_min     = -0.02
            suspicious_max = -0.10

        if score >= normal_min:
                      code = 0
elif score >= suspicious_max:
            code = 2
else:
            code = 1

    # == Random Forest path ==
elif model == "random_forest" and "random_forest" in models:
        clf = models["random_forest"]
        proba = clf.predict_proba(feat_vec)[0]
        pred_raw = int(clf.predict(feat_vec)[0])

        score = float(proba[1])  # probability of class 1 (suspicious)

        if pred_raw == 0:
                      code = 0
elif proba[1] > 0.65:
            code = 1
else:
            code = 2

else:
        raise ValueError(f"Unknown model '{model}' or model not loaded.")

    # Build per-hour label array (for bar chart colouring in UI)
      # Use the IF score for individual hours for richer visual feedback
    if model == "isolation_forest":
              clf = models["isolation_forest"]
else:
        clf = models.get("isolation_forest")

    hour_labels = []

    if clf is not None:
              for h in range(24):
                            h_feat = profile_to_features(profile.copy()).reshape(1, -1)
                            # For per-hour visual: just use overall score nudged by deviation from mean
                            dev = (profile[h] - profile.mean()) / (profile.std() + 1e-6)
            adj_score = score - abs(dev) * 0.05
            if adj_score >= (normal_min if _thresh else -0.02):
                              hour_labels.append("Normal")
elif adj_score >= (suspicious_max if _thresh else -0.10):
                hour_labels.append("Slightly Unusual")
else:
                hour_labels.append("Suspicious")
else:
        hour_labels = [LABEL_MAP[code]] * 24

    return {
              "label"      : LABEL_MAP[code],
              "code"       : code,
              "score"      : score,
              "hour_labels": hour_labels,
    }


def predict_batch(df: pd.DataFrame, model: str = "isolation_forest") -> pd.DataFrame:
      """Run predict_single on each row of a DataFrame with hourly columns 0..23."""
    results = []
    for _, row in df.iterrows():
              readings = [row[str(h)] if str(h) in row else row[h] for h in range(24)]
        results.append(predict_single(readings, model))
    return pd.DataFrame(results)
