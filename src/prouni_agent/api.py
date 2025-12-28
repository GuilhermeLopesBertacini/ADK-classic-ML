from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from prouni_agent.config import Paths
from prouni_agent.predict import load_model, predict_one

paths = Paths()
MODEL_PATH = os.getenv("PROUNI_MODEL_PATH", str(paths.default_model_path))

app = FastAPI(title="PROUNI Orientation API", version="0.1.0")

class PredictRequest(BaseModel):
    ano_concessao_bolsa: int = Field(..., alias="ANO_CONCESSAO_BOLSA")
    sexo_beneficiario: str | None = Field(None, alias="SEXO_BENEFICIARIO")
    raca_beneficiario: str | None = Field(None, alias="RACA_BENEFICIARIO")
    data_nascimento: str | None = Field(None, alias="DATA_NASCIMENTO")  # "YYYY-MM-DD"
    beneficiario_deficiente_fisico: str | None = Field(None, alias="BENEFICIARIO_DEFICIENTE_FISICO")
    regiao_beneficiario: str | None = Field(None, alias="REGIAO_BENEFICIARIO")
    uf_beneficiario: str | None = Field(None, alias="UF_BENEFICIARIO")
    municipio_beneficiario: str | None = Field(None, alias="MUNICIPIO_BENEFICIARIO")
    modalidade_ensino_bolsa: str | None = Field(None, alias="MODALIDADE_ENSINO_BOLSA")
    nome_curso_bolsa: str | None = Field(None, alias="NOME_CURSO_BOLSA")
    nome_turno_curso_bolsa: str | None = Field(None, alias="NOME_TURNO_CURSO_BOLSA")

class PredictResponse(BaseModel):
    label: str
    proba_integral: float | None
    proba_parcial: float | None

@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH}

@app.on_event("startup")
def _load():
    # Fail fast if model missing
    if not os.path.exists(MODEL_PATH):
        # allow running health without model? keep strict to avoid silent errors
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        model = load_model(MODEL_PATH)

        # Convert from aliases to expected training column names
        payload = {
            "ANO_CONCESSAO_BOLSA": req.ano_concessao_bolsa,
            "SEXO_BENEFICIARIO": req.sexo_beneficiario,
            "RACA_BENEFICIARIO": req.raca_beneficiario,
            "DATA_NASCIMENTO": req.data_nascimento,
            "BENEFICIARIO_DEFICIENTE_FISICO": req.beneficiario_deficiente_fisico,
            "REGIAO_BENEFICIARIO": req.regiao_beneficiario,
            "UF_BENEFICIARIO": req.uf_beneficiario,
            "MUNICIPIO_BENEFICIARIO": req.municipio_beneficiario,
            "MODALIDADE_ENSINO_BOLSA": req.modalidade_ensino_bolsa,
            "NOME_CURSO_BOLSA": req.nome_curso_bolsa,
            "NOME_TURNO_CURSO_BOLSA": req.nome_turno_curso_bolsa,
        }

        pred = predict_one(model, payload)
        return PredictResponse(
            label=pred.label,
            proba_integral=pred.proba_integral,
            proba_parcial=pred.proba_parcial,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))