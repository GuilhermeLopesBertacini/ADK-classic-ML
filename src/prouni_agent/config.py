from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True) # immutable
class Paths:
    project_root: Path = Path(".")
    default_data_csv: Path = Path("data") / "ProuniRelatorioDadosAbertos2020.csv"
    models_dir: Path = Path("models")
    default_model_path: Path = models_dir / "prouni_2020.joblib"