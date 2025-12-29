"""
API FastAPI para o Agente de Orientação PROUNI.

Arquitetura:
- Lifecycle management: modelo carregado no startup via PredictionService
- Routers separados: /api/v1 (técnico) e /adk (conversacional)
- CORS habilitado para integração com frontends
- Lógica centralizada em service.py (sem duplicação)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from prouni_agent.service import PredictionService, get_prediction_service
from prouni_agent.schemas import (
    PredictRequest,
    PredictResponse,
    HealthResponse,
    RootResponse,
    ADKPredictRequest,
    ADKPredictResponse,
    ADKHealthResponse,
)

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"
ADK_PREFIX = "/adk"


# ============================================================================
# ROUTER PRINCIPAL (/api/v1)
# ============================================================================

router = APIRouter(prefix=API_PREFIX, tags=["prouni"])


@router.get("/", response_model=RootResponse)
async def root():
    """Informações da API e endpoints disponíveis."""
    return RootResponse(
        message="PROUNI Orientation API - Classificação de bolsas (INTEGRAL vs PARCIAL)",
        version="1.0.0",
        endpoints={
            "health": f"{API_PREFIX}/health",
            "predict": f"{API_PREFIX}/predict",
            "adk_predict": f"{ADK_PREFIX}/predict-bolsa",
            "docs": f"{API_PREFIX}/docs",
        }
    )


@router.get("/health", response_model=HealthResponse)
async def health_check(service: PredictionService = Depends(get_prediction_service)):
    """Health check da API com status do modelo."""
    return HealthResponse(
        status="ok" if service.is_loaded else "error",
        model_path=str(service.model_path),
        model_loaded=service.is_loaded,
    )


@router.post("/predict", response_model=PredictResponse)
async def predict_bolsa(
    req: PredictRequest,
    service: PredictionService = Depends(get_prediction_service),
):
    """
    Predição de tipo de bolsa (INTEGRAL vs PARCIAL).
    
    Formato técnico com todas as colunas do dataset.
    Para integração conversacional (ADK), use /adk/predict-bolsa.
    """
    try:
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
        
        result = service.predict(payload)
        
        return PredictResponse(
            label=result.label,
            proba_integral=result.proba_integral,
            proba_parcial=result.proba_parcial,
        )
        
    except RuntimeError as e:
        logger.error(f"Modelo não carregado: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Erro na predição: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Erro ao processar: {str(e)}")


# ============================================================================
# ROUTER ADK (/adk) - Formato conversacional
# ============================================================================

adk_router = APIRouter(prefix=ADK_PREFIX, tags=["adk-integration"])


@adk_router.get("/health", response_model=ADKHealthResponse)
async def adk_health():
    """Health check específico para ADK."""
    return ADKHealthResponse(
        status="ok",
        service="PROUNI Orientation - ADK Integration",
        available_tools=["predict-bolsa"],
    )


@adk_router.post("/predict-bolsa", response_model=ADKPredictResponse)
async def predict_bolsa_adk(
    req: ADKPredictRequest,
    service: PredictionService = Depends(get_prediction_service),
):
    """
    Predição otimizada para Google ADK (function calling).
    
    Recebe parâmetros conversacionais (idade, curso, uf) e retorna
    resposta humanizada pronta para apresentação pelo agente.
    """
    try:
        result = service.predict_from_conversational(
            idade=req.idade,
            sexo=req.sexo,
            raca=req.raca,
            pcd=req.pcd,
            uf=req.uf,
            municipio=req.municipio,
            curso=req.curso,
            turno=req.turno,
            modalidade=req.modalidade,
        )
        
        return ADKPredictResponse(
            tipo_bolsa=result.label,
            probabilidade_integral=round(result.proba_integral * 100, 1),
            probabilidade_parcial=round(result.proba_parcial * 100, 1),
            confianca=result.confidence_level,
            mensagem=result.to_message(),
        )
        
    except RuntimeError as e:
        logger.error(f"Modelo não carregado: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Erro na predição ADK: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao processar: {str(e)}")


# ============================================================================
# LIFECYCLE MANAGEMENT
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia startup/shutdown da aplicação."""
    # === STARTUP ===
    logger.info("Iniciando PROUNI Orientation API...")
    
    service = get_prediction_service()
    logger.info(f"Caminho do modelo: {service.model_path}")
    
    if not service.model_exists:
        logger.error(f"Modelo não encontrado: {service.model_path}")
        raise RuntimeError("Execute 'make train' para gerar o modelo.")
    
    try:
        service.load_model()
        logger.info("Modelo carregado com sucesso")
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        raise
    
    logger.info("API iniciada com sucesso")
    yield
    
    # === SHUTDOWN ===
    logger.info("Finalizando API...")
    service.unload_model()
    logger.info("Modelo descarregado")


# ============================================================================
# APPLICATION FACTORY
# ============================================================================

def create_app() -> FastAPI:
    """Factory para criação da aplicação FastAPI."""
    application = FastAPI(
        title="PROUNI Orientation API",
        description=(
            "API para predição de tipo de bolsa (INTEGRAL vs PARCIAL) "
            "baseada em perfil socioeconômico. Modelo treinado com PROUNI 2020."
        ),
        version="1.0.0",
        docs_url=f"{API_PREFIX}/docs",
        redoc_url=f"{API_PREFIX}/redoc",
        openapi_url=f"{API_PREFIX}/openapi.json",
        lifespan=lifespan,
    )
    
    # CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Routers
    application.include_router(router)
    application.include_router(adk_router)
    
    logger.info("Aplicação FastAPI configurada")
    return application


# Instância global
app = create_app()


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PROUNI Orientation API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    
    logger.info(f"Docs: http://{args.host}:{args.port}{API_PREFIX}/docs")
    uvicorn.run("prouni_agent.api:app", host=args.host, port=args.port, reload=args.reload)
