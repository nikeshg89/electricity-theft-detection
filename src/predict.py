# -*- coding: utf-8 -*-
import os, json, joblib
import numpy as np
import pandas as pd

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
IF_PATH   = os.path.join(MODEL_DIR, "isolation_forest.pkl")
RF_PATH   = os.path.join(MODEL_DIR, "random_forest.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "thresholds.json")

LABEL_MAP = {0: "Normal Usage", 1: "Suspicious", 2: "Slightly Unusual"}
_models, _scaler, _thresh = {}, None, None

def load_models():
          global _models, _scaler, _thresh
          if not _models:
                        if not os.path.exists(IF_PATH): raise FileNotFoundError("Models missing")
                                      _models["isolation_forest"] = joblib.load(IF_PATH)
                        if os.path.exists(RF_PATH): _models["random_forest"] = joblib.load(RF_PATH)
                                      if os.path.exists(SCALER_PATH): _scaler = joblib.load(SCALER_PATH)
                                                    if os.path.exists(THRESHOLD_PATH):
                                                                      with open(THRESHOLD_PATH) as f: _thresh = json.load(f)
                                                                                return _models

def profile_to_features(p):
          p = np.array(p); h = np.arange(24); m = p.mean()
          n_mask = (h < 6) | (h >= 22); d_mask = ~n_mask
          n_sum = p[n_mask].sum(); d_sum = p[d_mask].sum()
          nd_ratio = n_sum / (d_sum + 1e-6)
          return np.array([m, p.std(), p.max(), p.min(), p.max()-p.min(), n_sum, d_sum, nd_ratio, float(np.argmax(p)), (p<0.1).sum()/24.0, (p>2*m+1e-6).sum()/24.0])

def predict_single(readings, model="isolation_forest"):
          load_models(); p = np.array(readings, dtype=float).clip(0)
          v = profile_to_features(p).reshape(1, -1)
          if _scaler: v = _scaler.transform(v)
                    if model == "isolation_forest":
                                  clf = _models["isolation_forest"]
                                  s = float(clf.decision_function(v)[0])
                                  n_m = _thresh.get("normal_min", -0.02) if _thresh else -0.02
                                  s_m = _thresh.get("suspicious_max", -0.10) if _thresh else -0.10
                                  if s >= n_m: c = 0
                    elif s >= s_m: c = 2
else: c = 1
    return {"label": LABEL_MAP[c], "code": c, "score": s, "hour_labels": ["Normal"]*24}

def predict_batch(df, model="isolation_forest"):
          results = []
    for _, row in df.iterrows():
                  readings = [row[h] for h in range(24)]
                  results.append(predict_single(readings, model))
              return pd.DataFrame(results)
