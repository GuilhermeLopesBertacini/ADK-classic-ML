from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import pandas as pd

from prouni_agent.features import add_age_feature, normalize_text_columns
from prouni_agent.modeling import ensure_columns

@dataclass(frozen=True)
class Prediction:
    label: str
    proba_integral: float | None
    proba_parcial: float | None

def load_model(model_path: str | Path):
    obj = joblib.load(model_path)
    return obj["pipeline"]

def predict_one(model, payload: dict) -> Prediction:
    df = pd.DataFrame([payload])

    # Align expected preprocessing steps
    df = add_age_feature(df)
    df = normalize_text_columns(df)
    df = ensure_columns(df)

    label = model.predict(df)[0]

    proba_integral = None
    proba_parcial = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0]
        classes = list(getattr(model, "classes_", []))
        if "INTEGRAL" in classes:
            proba_integral = float(proba[classes.index("INTEGRAL")])
        if "PARCIAL" in classes:
            proba_parcial = float(proba[classes.index("PARCIAL")])

    return Prediction(
        label=str(label),
        proba_integral=proba_integral,
        proba_parcial=proba_parcial,
    )