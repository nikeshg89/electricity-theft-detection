"""
preprocess.py
=============
Handles loading, cleaning, and normalizing the UCI Electricity Load Diagrams
dataset (LD2011_2014.txt).

Steps:
  1. Download the raw dataset if not already present.
    2. Parse the datetime index.
      3. Resample to hourly frequency.
        4. Handle missing values.
          5. Fit and apply StandardScaler.
            6. Save scaler for later inference.
            """

import os
import zipfile
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

# == Paths ==
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR   = os.path.join(BASE_DIR, "model")

RAW_ZIP  = os.path.join(DATASET_DIR, "LD2011_2014.txt.zip")
RAW_TXT  = os.path.join(DATASET_DIR, "LD2011_2014.txt")
PROC_CSV = os.path.join(DATASET_DIR, "processed_hourly.csv")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

UCI_URL = (
      "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
)

# == Download helpers ==

def download_dataset():
      """Download and unzip the UCI dataset if it does not already exist."""
      os.makedirs(DATASET_DIR, exist_ok=True)
      os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(RAW_TXT):
              print(f"[INFO] Raw dataset already exists: {RAW_TXT}")
              return

    if not os.path.exists(RAW_ZIP):
              print(f"[INFO] Downloading dataset from UCI ...")
              response = requests.get(UCI_URL, stream=True, timeout=120)
              total = int(response.headers.get("content-length", 0))
              with open(RAW_ZIP, "wb") as f, tqdm(
                            desc="Downloading", total=total, unit="B", unit_scale=True
              ) as bar:
                            for chunk in response.iter_content(chunk_size=8192):
                                              f.write(chunk)
                                              bar.update(len(chunk))
                                      print(f"[INFO] Download complete -> {RAW_ZIP}")

          print(f"[INFO] Extracting zip ...")
    with zipfile.ZipFile(RAW_ZIP, "r") as z:
              z.extractall(DATASET_DIR)
          print(f"[INFO] Extracted to {DATASET_DIR}")


# == Loading ==

def load_raw(nrows: int | None = None) -> pd.DataFrame:
      """
          Load the raw semicolon-separated file.

              Parameters
                  ----------
                      nrows : int, optional
                              Limit rows for quick prototyping (None = load all).

                                  Returns
                                      -------
                                          pd.DataFrame with datetime index and one column per client (MT_001...).
                                              """
      print(f"[INFO] Loading raw data (nrows={nrows}) ...")
      df = pd.read_csv(
          RAW_TXT,
          sep=";",
          decimal=",",        # European locale: comma as decimal separator
          index_col=0,
          parse_dates=True,
          nrows=nrows,
          low_memory=False,
      )
      df.index = pd.to_datetime(df.index)
      df.index.name = "datetime"
      # Replace 0 with NaN - genuine zeros are rare, usually means missing reading
      df.replace(0.0, np.nan, inplace=True)
      print(f"[INFO] Loaded shape: {df.shape}")
      return df



# == Resampling ==

def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
      """Resample 15-minute data to hourly sums (kWh)."""
      print("[INFO] Resampling to hourly frequency ...")
      df_hourly = df.resample("h").sum(min_count=1)
      return df_hourly


# == Missing value handling ==

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
      """Forward-fill then backward-fill residual NaNs; drop all-NaN columns."""
      before = df.isna().sum().sum()
      df = df.ffill().bfill()
      after = df.isna().sum().sum()
      print(f"[INFO] Missing values: {before} -> {after} (after ffill/bfill)")
      # Drop clients that are still entirely NaN
      df.dropna(axis=1, how="all", inplace=True)
    return df


# == Normalisation ==

def scale_data(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
      """
          Apply StandardScaler column-wise (per client).

              Parameters
                  ----------
                      fit : bool
                              If True, fit a new scaler and save it.  If False, load the saved scaler.
                                  """
      if fit:
                scaler = StandardScaler()
                scaled = scaler.fit_transform(df)
                joblib.dump(scaler, SCALER_PATH)
                print(f"[INFO] Scaler fitted and saved -> {SCALER_PATH}")
else:
        scaler = joblib.load(SCALER_PATH)
          scaled = scaler.transform(df)
        print(f"[INFO] Scaler loaded from {SCALER_PATH}")

    df_scaled = pd.DataFrame(scaled, index=df.index, columns=df.columns)
    return df_scaled


# == Main pipeline ==

def run_preprocessing(nrows: int | None = None, force: bool = False) -> pd.DataFrame:
      """
          Full preprocessing pipeline.

              Returns
                  -------
                      Hourly, cleaned, scaled DataFrame saved as processed_hourly.csv.
                          """
    if os.path.exists(PROC_CSV) and not force:
              print(f"[INFO] Processed file already exists, loading -> {PROC_CSV}")
              df = pd.read_csv(PROC_CSV, index_col=0, parse_dates=True)
              return df

    download_dataset()
    df = load_raw(nrows=nrows)
    df = resample_hourly(df)
    df = handle_missing(df)
    df_scaled = scale_data(df, fit=True)
    df_scaled.to_csv(PROC_CSV)
    print(f"[INFO] Processed data saved -> {PROC_CSV}  shape={df_scaled.shape}")
    return df_scaled


if __name__ == "__main__":
      run_preprocessing()
