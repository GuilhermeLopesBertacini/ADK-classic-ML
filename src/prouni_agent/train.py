from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from prouni_agent.data import read_prouni_csv, basic_clean
from prouni_agent.features import add_age_feature, normalize_text_columns, make_xy
from prouni_agent.modeling import build_pipeline, ensure_columns

def train(data_path: Path, out_path: Path) -> None:
    df = read_prouni_csv(data_path)
    df = basic_clean(df)
    df = add_age_feature(df)
    df = normalize_text_columns(df)

    X, y = make_xy(df)

    # remove rows without target
    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    # remove rows without minimal features
    X = ensure_columns(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    # AUC only if binary and probabilities exist
    if hasattr(pipe, "predict_proba") and len(pipe.classes_) == 2:
        proba = pipe.predict_proba(X_test)
        # positive class = INTEGRAL (if present)
        classes = list(pipe.classes_)
        if "INTEGRAL" in classes:
            pos_idx = classes.index("INTEGRAL")
            auc = roc_auc_score((y_test == "INTEGRAL").astype(int), proba[:, pos_idx])
            print(f"\nROC-AUC (INTEGRAL as positive): {auc:.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipe,
            "classes": getattr(pipe, "classes_", None),
            "feature_columns": list(X.columns),
        },
        out_path,
    )
    print(f"\nSaved model to: {out_path}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to CSV")
    ap.add_argument("--out", type=str, required=True, help="Output .joblib path")
    args = ap.parse_args()
    train(Path(args.data), Path(args.out))

if __name__ == "__main__":
    main()