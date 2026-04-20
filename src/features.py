"""
features.py
===========
Derive time-based and statistical features from the hourly consumption data.

Features computed PER CLIENT:
  - hour            : hour of the day (0-23)
    - day_of_week     : 0=Monday ... 6=Sunday
      - is_weekend      : 1 if Sat/Sun
        - month           : 1-12
          - rolling_mean_24h: 24-hour rolling mean
            - rolling_std_24h : 24-hour rolling std
              - consumption     : raw (normalised) reading
                - prev_day_diff   : difference from same hour 24 steps ago

                The output is a flat feature matrix used by all ML models.
                """

import paths
import pandas as pd # MISSING IMPORT ADDED

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
      """Add calendar features shared across all clients."""
      meta = pd.DataFrame(index=df.index)
      meta["hour"]        = df.index.hour
      meta["day_of_week"] = df.index.dayofweek
      meta["is_weekend"]  = (df.index.dayofweek >= 5).astype(int)
      meta["month"]       = df.index.month
      return meta


def client_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
      """
          For each timestamp, compute cross-client aggregate features:
                - mean_consumption : average across all clients
                      - std_consumption  : std across all clients
                            - max_consumption  : maximum across all clients
                                  - min_consumption  : minimum across all clients

                                      These represent the "grid load" context useful for anomaly detection.
                                          """
      agg = pd.DataFrame(index=df.index)
      agg["mean_consumption"] = df.mean(axis=1)
      agg["std_consumption"]  = df.std(axis=1)
      agg["max_consumption"]  = df.max(axis=1)
      agg["min_consumption"]  = df.min(axis=1)
      return agg


def rolling_features(series: pd.Series, window: int = 24) -> pd.DataFrame:
      """Compute rolling statistics for a single client's time series."""
      feats = pd.DataFrame(index=series.index)
      feats[f"rolling_mean_{window}h"] = series.rolling(window, min_periods=1).mean()
      feats[f"rolling_std_{window}h"]  = series.rolling(window, min_periods=1).std().fillna(0)
      feats["prev_day_diff"]           = series.diff(window).fillna(0)
      return feats


def build_feature_matrix(
      df_scaled: pd.DataFrame,
      sample_clients: int | None = 50,
) -> pd.DataFrame:
      """
          Build the full model-ready feature matrix.

              Parameters
                  ----------
                      df_scaled      : Scaled hourly DataFrame (rows=time, cols=clients).
                          sample_clients : Use first N clients to keep memory manageable.
                                               Pass None to use all clients.

                                                   Returns
                                                       -------
                                                           feature_df : DataFrame with one row per (client, timestamp) and
                                                                            all engineered features.
                                                                                """
      print(f"[INFO] Building feature matrix ...")
      clients = df_scaled.columns[:sample_clients] if sample_clients else df_scaled.columns

    time_meta = extract_time_features(df_scaled)
    agg_feats = client_aggregate_features(df_scaled)

    records = []
    for i, client in enumerate(clients):
              if i % 10 == 0:
                            print(f"  ... processing {i+1}/{len(clients)} clients")
                        series   = df_scaled[client]
        roll     = rolling_features(series, window=24)
        client_df = pd.DataFrame({
                      "client"         : client,
                      "consumption"    : series.values,
        }, index=df_scaled.index)
        client_df = client_df.join(time_meta).join(agg_feats).join(roll)
        records.append(client_df)

    feature_df = pd.concat(records, axis=0).reset_index()
    feature_df.rename(columns={"index": "datetime"}, inplace=True)
    # Drop rows with any NaN
    feature_df.dropna(inplace=True)
    print(f"[INFO] Feature matrix shape: {feature_df.shape}")
    return feature_df


def save_features(feature_df: pd.DataFrame): # MISSING FUNCTION ADDED
      feature_df.to_csv(paths.FEAT_CSV, index=False)
    print(f"[INFO] Features saved -> {paths.FEAT_CSV}")


def load_features() -> pd.DataFrame:
      df = pd.read_csv(paths.FEAT_CSV, parse_dates=["datetime"])
    print(f"[INFO] Features loaded from {paths.FEAT_CSV}  shape={df.shape}")
    return df


if __name__ == "__main__":
      from preprocess import run_preprocessing
    df_scaled = run_preprocessing()
    feat_df   = build_feature_matrix(df_scaled)
    save_features(feat_df)
