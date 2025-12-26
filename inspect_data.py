from pathlib import Path
import pandas as pd

def inspect_data(df: pd.DataFrame) -> None:
    """
    Inspeciona um DataFrame do pandas e imprime informações úteis sobre ele.

    Args:
        df (pd.DataFrame): DataFrame a ser inspecionado
    """
    print(f"Shape: {df.shape}")

    print(f"\nColumns:")
    for c in df.columns:
        print(f" - {c}")

    print(f"\n dtypes: \n{df.dtypes.head(10)}")

    print(f"\n Missing values: \n{(df.isna().mean().sort_values(ascending=False) * 100).round(2).head(10)}")

    if "TIPO_BOLSA" in df.columns:
        print("\nTIPO_BOLSA value_counts:\n" + df["TIPO_BOLSA"].astype("string").value_counts(dropna=False).to_string())

    print(f"\n Sample data: \n{df.head(30)}")


def main() -> None:
    csv_path = Path("data") / "ProuniRelatorioDadosAbertos2020.csv"
    try:
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    inspect_data(df)

if __name__ == "__main__":
    main()