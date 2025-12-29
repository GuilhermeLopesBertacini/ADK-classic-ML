from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "ANO_CONCESSAO_BOLSA",
    "TIPO_BOLSA",
    "SEXO_BENEFICIARIO",
    "RACA_BENEFICIARIO",
    "DATA_NASCIMENTO",
    "BENEFICIARIO_DEFICIENTE_FISICO",
    "REGIAO_BENEFICIARIO",
    "UF_BENEFICIARIO",
    "MUNICIPIO_BENEFICIARIO",
    "MODALIDADE_ENSINO_BOLSA",
    "NOME_CURSO_BOLSA",
    "NOME_TURNO_CURSO_BOLSA",
]

def read_prouni_csv(path: str| Path) -> pd.DataFrame:
    """Lê o CSV do Prouni e retorna um DataFrame do pandas.

    Args:
        path (str | Path): Caminho para o arquivo CSV.

    Returns:
        pd.DataFrame: DataFrame contendo os dados do Prouni.
    """
    path = Path(path)
    try:
        return pd.read_csv(path, sep=";", encoding="utf-8")
    except UnicodeDecodeError as e:
        print(f"Erro ao ler o arquivo com utf-8: {e}. Tentando utf-8-sig...")
        return pd.read_csv(path, sep=";", encoding="utf-8-sig")


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Realiza uma limpeza básica no DataFrame do Prouni.

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: DataFrame limpo.
    """
    df = df.copy()

    # Garantir que todas as colunas necessárias estão presentes
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltando colunas obrigatórias: {missing_cols}")

    # reduzir para colunas necessárias apenas
    df = df[REQUIRED_COLUMNS].copy()

    # normalizar strings vazias para NA
    # inclui somente colunas que representam vetores (string, list, dict)
    obj_cols = df.select_dtypes(include=["object"]).columns.to_list()
    for col in obj_cols:
        df[col] = df[col].astype("string").str.strip()
        df.loc[df[col].isin(["", "NA", "N/A", "NULL", "None"]), col] = pd.NA
        # Converter para object dtype (np.nan) para compatibilidade com sklearn
        df[col] = df[col].astype("object").fillna(np.nan)

    return df