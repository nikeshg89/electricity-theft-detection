"""
run_pipeline.py
===============
One-command runner that executes the full ML pipeline:

  Step 1 -> Data preprocessing  (src/preprocess.py)
  Step 2 -> Feature engineering  (src/features.py)
  Step 3 -> Model training        (src/train.py)

Run:
    python run_pipeline.py
"""

import sys, os, time

# Force UTF-8 output (avoids Windows cp1252 encoding errors with emoji/unicode)
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add src to path
SRC = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC)

BANNER = """
==========================================================
  Smart Electricity Theft Detection - Full Pipeline
  Isolation Forest | LOF | Random Forest
==========================================================
"""

SEP = "-" * 58


def step(n: int, title: str):
    print(f"\n{SEP}")
    print(f"  STEP {n}: {title}")
    print(SEP)


if __name__ == "__main__":
    print(BANNER)
    t0 = time.time()

    # Step 1: Preprocessing
    step(1, "Data Preprocessing")
    from preprocess import run_preprocessing
    df_scaled = run_preprocessing()

    # Step 2: Feature Engineering
    step(2, "Feature Engineering")
    from features import build_feature_matrix, save_features, load_features, FEAT_CSV

    if os.path.exists(FEAT_CSV):
        print(f"[INFO] Features already exist, loading ...")
        feat_df = load_features()
    else:
        feat_df = build_feature_matrix(df_scaled, sample_clients=50)
        save_features(feat_df)

    # Step 3: Model Training
    step(3, "Model Training & Evaluation")
    from train import run_training
    run_training()

    elapsed = time.time() - t0
    print(f"\n{SEP}")
    print(f"  [OK] Pipeline complete in {elapsed/60:.1f} minutes")
    print(f"  Launch the app: streamlit run app.py")
    print(f"{SEP}\n")
