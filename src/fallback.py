# -*- coding: utf-8 -*-
"""
fallback.py
===========
Generates synthetic fallback data if the massive UCI dataset is unavailable.
This ensures the app can quickly train its models and run instantly for demos.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
FEAT_CSV = os.path.join(DATASET_DIR, "features.csv")
PROC_CSV = os.path.join(DATASET_DIR, "processed_hourly.csv")

def generate_fallback_data(num_clients=50, days=30):
      os.makedirs(DATASET_DIR, exist_ok=True)
      print(f"[INFO] Generating synthetic fallback dataset for {num_clients} clients over {days} days...")

    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(days * 24)]

    records = []

    # Generate Normal clients (95%)
    normal_clients = int(num_clients * 0.95)
    suspicious_clients = num_clients - normal_clients

    clients = []
    for i in range(normal_clients):
              clients.append({"id": f"MT_{i:03d}", "type": "normal"})
          for i in range(suspicious_clients):
                    clients.append({"id": f"MT_{normal_clients+i:03d}", "type": "suspicious"})

    for ts in timestamps:
              hour = ts.hour
              day_of_week = ts.weekday()
              is_weekend = 1 if day_of_week >= 5 else 0
              month = ts.month

        # Peak smooth curve around hour 13
              grid_base = 3.0 + 2.0 * np.sin(np.pi * (hour - 3) / 12) if 3 <= hour <= 15 else 3.0 + 2.0 * np.sin(np.pi * (hour - 15) / 12)
              grid_base = max(1.0, grid_base)

        for client in clients:
                      if client["type"] == "normal":
                                        # Normal pattern: smooth low noise
                                        cons = grid_base * np.random.normal(1.0, 0.1) + np.random.uniform(0, 0.2)
else:
                  # Suspicious: randomized severe spikes at odd hours or flatlines
                  if np.random.rand() > 0.5:
                                        cons = np.random.uniform(0.01, 0.05) # meter tamper
else:
                    if hour < 6 or hour > 22:
                                              cons = grid_base * np.random.normal(4.0, 1.0) # night time heavy theft
                    else:
                                              cons = grid_base * np.random.normal(0.5, 0.1)

              cons = max(0.01, cons) # No exact zeros

            records.append({
                              "datetime": ts,
                              "client": client["id"],
                              "consumption": cons,
                              "hour": hour,
                              "day_of_week": day_of_week,
                              "is_weekend": is_weekend,
                              "month": month,
            })

    df = pd.DataFrame(records)

    # Calculate rolling and grid-level features
    print("[INFO] Computing aggregate and rolling features for synthetic data...")

    # Cross-client aggregates per timestamp
    agg = df.groupby("datetime")["consumption"].agg(["mean", "std", "max", "min"]).reset_index()
    agg.rename(columns={"mean": "mean_consumption", "std": "std_consumption", "max": "max_consumption", "min": "min_consumption"}, inplace=True)
    df = df.merge(agg, on="datetime", how="left")

    # Rolling features per client
    df = df.sort_values(by=["client", "datetime"])
    df["rolling_mean_24h"] = df.groupby("client")["consumption"].transform(lambda x: x.rolling(24, min_periods=1).mean())
    df["rolling_std_24h"] = df.groupby("client")["consumption"].transform(lambda x: x.rolling(24, min_periods=1).std().fillna(0))
    df["prev_day_diff"] = df.groupby("client")["consumption"].transform(lambda x: x.diff(24).fillna(0))

    # Feature 1: max/min diff
    df["max_min_diff"] = df["max_consumption"] - df["min_consumption"]

    # Feature 2: Night vs Day ratio over rolling 24h
    df['is_night'] = df['hour'].apply(lambda h: 1 if (h < 6 or h >= 18) else 0)
    df['is_day'] = 1 - df['is_night']
    df['night_cons'] = df['consumption'] * df['is_night']
    df['day_cons'] = df['consumption'] * df['is_day']

    night_roll = df.groupby("client")["night_cons"].transform(lambda x: x.rolling(24, min_periods=1).sum())
    day_roll = df.groupby("client")["day_cons"].transform(lambda x: x.rolling(24, min_periods=1).sum())

    df["night_vs_day_ratio"] = night_roll / (day_roll + 1e-4) # Avoid zero div

    # Ensure correct column order
    FEATURE_COLS = [
              "datetime", "client", "consumption", "hour", "day_of_week", "is_weekend", "month",
              "mean_consumption", "std_consumption", "max_consumption", "min_consumption",
              "rolling_mean_24h", "rolling_std_24h", "prev_day_diff", "max_min_diff", "night_vs_day_ratio"
    ]
    df = df[FEATURE_COLS]

    df.to_csv(FEAT_CSV, index=False)
    print(f"[INFO] Synthetic data saved -> {FEAT_CSV} (Shape: {df.shape})")

    # Also save a dummy processed_hourly.csv so app doesn't complain
    dummy_proc = df.pivot(index="datetime", columns="client", values="consumption")
    dummy_proc.to_csv(PROC_CSV)

    return True

if __name__ == "__main__":
      generate_fallback_data()
  
                              
