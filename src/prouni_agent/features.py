from __future__ import annotations # type hint in runtime

import re
import unicodedata
import pandas as pd

TEXT_COLS = [
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

def _normalize_text(s: str) -> str:
    """Normaliza uma string removendo acentuação e convertendo para maiúsculas.

    Args:
        s (str): String a ser normalizada.

    Returns:
        str: String normalizada.
    """
    s = s.strip().upper()

    # decomposição e remoção de á para a + '
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    s = re.sub(r"[^A-Z0-9 _\-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def add_age_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse data nascimento
    dt = pd.to_datetime(df["DATA_NASCIMENTO"], errors="coerce", dayfirst=False)
    birth_year = dt.dt.year
    df["IDADE"] = (df["ANO_CONCESSAO_BOLSA"].astype("Int64") - birth_year.astype("Int64")).astype("Int64")
    df.loc[(df["IDADE"] < 14) | (df["IDADE"] > 80), "IDADE"] = pd.NA
    return df

def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza colunas de texto, removendo acentuação e convertendo para maiúsculas.

    Args:
        df (pd.DataFrame): DataFrame com colunas de texto a serem normalizadas.

    Returns:
        pd.DataFrame: DataFrame com colunas de texto normalizadas.
    """
    df = df.copy()
    for col in TEXT_COLS:
        df[col] = df[col].astype("string")
        df[col] = df[col].map(lambda x: _normalize_text(x) if isinstance(x, str) else x)
    return df

def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa o DataFrame em features (X) e target (y).

    Args:
        df (pd.DataFrame): DataFrame contendo as features e o target.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Tupla contendo o DataFrame de features (X) e a Series do target (y).
    """
    df = df.copy()

    y_raw = df["TIPO_BOLSA"].astype("string").fillna(pd.NA)

    def map_target(v: str | pd.NA):
        if v is pd.NA:
            return pd.NA
        v2 = _normalize_text(str(v))
        if "INTEGRAL" in v2:
            return "INTEGRAL"
        if "PARCIAL" in v2:
            return "PARCIAL"
        return pd.NA
    
    y = y_raw.map(map_target)

    x = df.drop(columns=["TIPO_BOLSA"])

    return x, y