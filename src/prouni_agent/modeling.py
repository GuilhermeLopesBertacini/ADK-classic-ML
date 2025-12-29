from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
# fill none values
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL_FEATURES = [
    "SEXO_BENEFICIARIO",
    "RACA_BENEFICIARIO",
    "BENEFICIARIO_DEFICIENTE_FISICO",
    "REGIAO_BENEFICIARIO",
    "UF_BENEFICIARIO",
    "MUNICIPIO_BENEFICIARIO",
    "MODALIDADE_ENSINO_BOLSA",
    "NOME_CURSO_BOLSA",
    "NOME_TURNO_CURSO_BOLSA",
]

NUMERIC_FEATURES = [
    "ANO_CONCESSAO_BOLSA",
    "IDADE",
]

def build_pipeline() -> Pipeline:
    """Constrói o pipeline de modelagem.

    Returns:
        Pipeline: Pipeline de modelagem.
    """

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=10))
        ]
    )

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
            ("num", num_pipe, NUMERIC_FEATURES)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=None, # sklearn uses liblinear default
        # altera o peso das classes para lidar com desbalanceamento
        class_weight="balanced",
        solver="lbfgs"
    )

    pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", clf)
        ]
    )

    return pipe


def ensure_columns(X: pd.DataFrame) -> pd.DataFrame:
    expected = set(CATEGORICAL_FEATURES + NUMERIC_FEATURES)
    missing = sorted(expected - set(X.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return X