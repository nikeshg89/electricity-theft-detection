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

import paths

UCI_URL = (
      "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
)

# -- Download helpers ---------------------------------------------------------

def download_dataset():
      """Download and unzip the UCI dataset if it does not already exist."""
      paths.DATASET_DIR.mkdir(parents=True, exist_ok=True)
      paths.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if paths.RAW_TXT.exists():
              print(f"[INFO] Raw dataset already exists: {paths.RAW_TXT}")
              return

    if not paths.RAW_ZIP.exists():
              print(f"[INFO] Downloading dataset from UCI ...")
              response = requests.get(UCI_URL, stream=True, timeout=120)
              total = int(response.headers.get("content-length", 0))
              with open(paths.RAW_ZIP, "wb") as f, tqdm(
                            desc="Downloading", total=total, unit="B", unit_scale=True
              ) as bar:
                            for chunk in response.iter_content(chunk_size=8192):
                                              f.write(chunk)
                                              bar.update(len(chunk))
                                      print(f"[INFO] Download complete -> {paths.RAW_ZIP}")


    print(f"[INFO] Extracting zip ...")
    with zipfile.ZipFile(paths.RAW_ZIP, "r") as z:
              z.extractall(paths.DATASET_DIR)
          print(f"[INFO] Extracted to {paths.DATASET_DIR}")


# -- Loading ------------------------------------------------------------------

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
          paths.RAW_TXT,
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


# -- Resampling ----------------------------------------------------------------

def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
      """Resample 15-minute data to hourly sums (kWh)."""
      print("[INFO] Resampling to hourly frequency ...")
      df_hourly = df.resample("h").sum(min_count=1)
      return df_hourly


# -- Missing value handling ----------------------------------------------------

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
      """Forward-fill then backward-fill residual NaNs; drop all-NaN columns."""
      before = df.isna().sum().sum()
      df = df.ffill().bfill()
      after = df.isna().sum().sum()
      print(f"[INFO] Missing values: {before} -> {after} (after ffill/bfill)")
      # Drop clients that are still entirely NaN
      df.dropna(axis=1, how="all", inplace=True)
      return df



# -- Normalisation -------------------------------------------------------------

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
                joblib.dump(scaler, paths.SCALER_PATH)
                print(f"[INFO] Scaler fitted and saved -> {paths.SCALER_PATH}")
else:
        scaler = joblib.load(paths.SCALER_PATH)
          scaled = scaler.transform(df)
        print(f"[INFO] Scaler loaded from {paths.SCALER_PATH}")

    df_scaled = pd.DataFrame(scaled, index=df.index, columns=df.columns)
    return df_scaled


# -- Main pipeline -------------------------------------------------------------

def run_preprocessing(nrows: int | None = None, force: bool = False) -> pd.DataFrame:
      """
          Full preprocessing pipeline.

              Returns
                  -------
                      Hourly, cleaned, scaled DataFrame saved as processed_hourly.csv.
                          """
    if paths.PROC_CSV.exists() and not force:
              print(f"[INFO] Processed file already exists, loading -> {paths.PROC_CSV}")
              df = pd.read_csv(paths.PROC_CSV, index_col=0, parse_dates=True)
              return df

    download_dataset()
    df = load_raw(nrows=nrows)
    df = resample_hourly(df)
    df = handle_missing(df)
    df_scaled = scale_data(df, fit=True)
    df_scaled.to_csv(paths.PROC_CSV)
    print(f"[INFO] Processed data saved -> {paths.PROC_CSV}  shape={df_scaled.shape}")
    return df_scaled


if __name__ == "__main__":
      run_preprocessing()
