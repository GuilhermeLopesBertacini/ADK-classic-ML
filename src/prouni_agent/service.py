"""
Camada de serviço para predição de bolsas PROUNI.

Centraliza toda a lógica de negócio, evitando duplicação entre endpoints.
Os controllers (api.py, adk.py) apenas validam input e formatam output.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from prouni_agent.config import Paths
from prouni_agent.features import add_age_feature, normalize_text_columns
from prouni_agent.modeling import ensure_columns
import pandas as pd


@dataclass(frozen=True)
class PredictionResult:
    """Resultado padronizado de uma predição."""
    label: str
    proba_integral: float
    proba_parcial: float
    
    @property
    def confidence_level(self) -> str:
        """Calcula nível de confiança baseado na diferença entre probabilidades."""
        diff = abs(self.proba_integral - self.proba_parcial) * 100
        if diff > 40:
            return "ALTA"
        elif diff > 20:
            return "MÉDIA"
        return "BAIXA"
    
    def to_message(self, lang: str = "pt-BR") -> str:
        """Gera mensagem humanizada para apresentação."""
        prob = self.proba_integral if self.label == "INTEGRAL" else self.proba_parcial
        tipo = "INTEGRAL (100%)" if self.label == "INTEGRAL" else "PARCIAL (50%)"
        return (
            f"Com base no seu perfil, você tem {prob * 100:.1f}% de chance de "
            f"conseguir uma bolsa {tipo} no PROUNI. "
            f"A confiança desta predição é {self.confidence_level.lower()}."
        )


class PredictionService:
    """
    Serviço singleton para predição de bolsas.
    
    Encapsula carregamento do modelo e lógica de predição.
    Projetado para ser injetado nos endpoints via dependency injection.
    """
    
    _instance: "PredictionService | None" = None
    _model: Any = None
    
    def __new__(cls) -> "PredictionService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def model_path(self) -> Path:
        """Caminho do modelo (configurável via env var)."""
        import os
        paths = Paths()
        return Path(os.getenv("PROUNI_MODEL_PATH", str(paths.default_model_path)))
    
    @property
    def is_loaded(self) -> bool:
        """Verifica se modelo está carregado."""
        return self._model is not None
    
    @property
    def model_exists(self) -> bool:
        """Verifica se arquivo do modelo existe."""
        return self.model_path.exists()
    
    def load_model(self) -> None:
        """Carrega modelo do disco para memória."""
        if not self.model_exists:
            raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
        
        bundle = joblib.load(self.model_path)
        self._model = bundle["pipeline"]
    
    def unload_model(self) -> None:
        """Descarrega modelo da memória."""
        self._model = None
    
    def predict(self, payload: dict[str, Any]) -> PredictionResult:
        """
        Executa predição com payload no formato do modelo.
        
        Args:
            payload: Dicionário com colunas do dataset
                     (ANO_CONCESSAO_BOLSA, SEXO_BENEFICIARIO, etc)
        
        Returns:
            PredictionResult com label e probabilidades
        
        Raises:
            RuntimeError: Se modelo não estiver carregado
            ValueError: Se payload inválido
        """
        if not self.is_loaded:
            raise RuntimeError("Modelo não carregado. Chame load_model() primeiro.")
        
        # Prepara DataFrame
        df = pd.DataFrame([payload])
        df = add_age_feature(df)
        df = normalize_text_columns(df)
        df = ensure_columns(df)
        
        # Predição
        label = self._model.predict(df)[0]
        
        # Probabilidades
        proba_integral = 0.5
        proba_parcial = 0.5
        
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(df)[0]
            classes = list(getattr(self._model, "classes_", []))
            if "INTEGRAL" in classes:
                proba_integral = float(proba[classes.index("INTEGRAL")])
            if "PARCIAL" in classes:
                proba_parcial = float(proba[classes.index("PARCIAL")])
        
        return PredictionResult(
            label=str(label),
            proba_integral=proba_integral,
            proba_parcial=proba_parcial,
        )
    
    def predict_from_conversational(
        self,
        idade: int | None = None,
        sexo: str | None = None,
        raca: str | None = None,
        pcd: bool | None = None,
        uf: str | None = None,
        municipio: str | None = None,
        curso: str | None = None,
        turno: str | None = None,
        modalidade: str | None = None,
    ) -> PredictionResult:
        """
        Predição com parâmetros conversacionais (para integração ADK).
        
        Converte automaticamente parâmetros simplificados para o formato
        esperado pelo modelo.
        """
        # Ano base do modelo
        ano_base = 2020
        
        # Calcula data de nascimento aproximada
        data_nascimento = None
        if idade is not None:
            ano_nascimento = ano_base - idade
            data_nascimento = f"{ano_nascimento}-01-01"
        
        # Monta payload no formato do modelo
        payload = {
            "ANO_CONCESSAO_BOLSA": ano_base,
            "SEXO_BENEFICIARIO": sexo,
            "RACA_BENEFICIARIO": raca,
            "DATA_NASCIMENTO": data_nascimento,
            "BENEFICIARIO_DEFICIENTE_FISICO": "S" if pcd else ("N" if pcd is False else None),
            "REGIAO_BENEFICIARIO": self._infer_region(uf),
            "UF_BENEFICIARIO": uf,
            "MUNICIPIO_BENEFICIARIO": municipio,
            "MODALIDADE_ENSINO_BOLSA": modalidade,
            "NOME_CURSO_BOLSA": curso,
            "NOME_TURNO_CURSO_BOLSA": turno,
        }
        
        return self.predict(payload)
    
    @staticmethod
    def _infer_region(uf: str | None) -> str | None:
        """Infere região a partir da UF."""
        if not uf:
            return None
        
        uf_to_region = {
            # Norte
            "AC": "NORTE", "AP": "NORTE", "AM": "NORTE", "PA": "NORTE",
            "RO": "NORTE", "RR": "NORTE", "TO": "NORTE",
            # Nordeste
            "AL": "NORDESTE", "BA": "NORDESTE", "CE": "NORDESTE", "MA": "NORDESTE",
            "PB": "NORDESTE", "PE": "NORDESTE", "PI": "NORDESTE", "RN": "NORDESTE", "SE": "NORDESTE",
            # Centro-Oeste
            "DF": "CENTRO-OESTE", "GO": "CENTRO-OESTE", "MT": "CENTRO-OESTE", "MS": "CENTRO-OESTE",
            # Sudeste
            "ES": "SUDESTE", "MG": "SUDESTE", "RJ": "SUDESTE", "SP": "SUDESTE",
            # Sul
            "PR": "SUL", "RS": "SUL", "SC": "SUL",
        }
        return uf_to_region.get(uf.upper())


# INSTÂNCIA GLOBAL (SINGLETON)

def get_prediction_service() -> PredictionService:
    """Dependency injection para FastAPI."""
    return PredictionService()
