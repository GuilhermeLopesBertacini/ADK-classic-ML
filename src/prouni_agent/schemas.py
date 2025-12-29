"""
Schemas Pydantic para validação de entrada/saída da API.

Este módulo centraliza todos os modelos de dados (DTOs - Data Transfer Objects)
utilizados nos endpoints da API, seguindo o padrão de separação de responsabilidades.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# SCHEMAS DA API PRINCIPAL (/api/v1)

class PredictRequest(BaseModel):
    """Schema de entrada para predição de tipo de bolsa (formato técnico)."""
    ano_concessao_bolsa: int = Field(
        ...,
        description="Ano de concessão da bolsa (ex: 2020)",
        ge=2000,
        le=2030
    )
    sexo_beneficiario: str | None = Field(
        None,
        description="Sexo do beneficiário (M/F)"
    )
    raca_beneficiario: str | None = Field(
        None,
        description="Raça/cor declarada pelo beneficiário"
    )
    data_nascimento: str | None = Field(
        None,
        description="Data de nascimento (YYYY-MM-DD ou DD/MM/YYYY)"
    )
    beneficiario_deficiente_fisico: str | None = Field(
        None,
        description="Beneficiário é PCD? (S/N)"
    )
    regiao_beneficiario: str | None = Field(
        None,
        description="Região do beneficiário (NORTE, NORDESTE, etc)"
    )
    uf_beneficiario: str | None = Field(
        None,
        description="UF do beneficiário (SP, RJ, etc)"
    )
    municipio_beneficiario: str | None = Field(
        None,
        description="Município do beneficiário"
    )
    modalidade_ensino_bolsa: str | None = Field(
        None,
        description="Modalidade (PRESENCIAL, EAD, etc)"
    )
    nome_curso_bolsa: str | None = Field(
        None,
        description="Nome do curso (DIREITO, MEDICINA, etc)"
    )
    nome_turno_curso_bolsa: str | None = Field(
        None,
        description="Turno (MATUTINO, NOTURNO, etc)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "ano_concessao_bolsa": 2020,
                "sexo_beneficiario": "F",
                "raca_beneficiario": "PARDA",
                "data_nascimento": "2004-01-15",
                "beneficiario_deficiente_fisico": "N",
                "regiao_beneficiario": "SUDESTE",
                "uf_beneficiario": "SP",
                "municipio_beneficiario": "SAO PAULO",
                "modalidade_ensino_bolsa": "PRESENCIAL",
                "nome_curso_bolsa": "DIREITO",
                "nome_turno_curso_bolsa": "NOTURNO"
            }
        }


class PredictResponse(BaseModel):
    """Schema de saída da predição (formato técnico)."""
    label: str = Field(..., description="Classe predita (INTEGRAL ou PARCIAL)")
    proba_integral: float | None = Field(
        None,
        description="Probabilidade de bolsa integral (0.0 a 1.0)",
        ge=0.0,
        le=1.0
    )
    proba_parcial: float | None = Field(
        None,
        description="Probabilidade de bolsa parcial (0.0 a 1.0)",
        ge=0.0,
        le=1.0
    )


class HealthResponse(BaseModel):
    """Schema de resposta do health check."""
    status: str = Field(..., description="Status da API (ok ou error)")
    model_path: str = Field(..., description="Caminho do modelo carregado")
    model_loaded: bool = Field(..., description="Modelo carregado com sucesso?")


class RootResponse(BaseModel):
    """Schema de resposta do endpoint raiz."""
    message: str = Field(..., description="Mensagem de boas-vindas")
    version: str = Field(..., description="Versão da API")
    endpoints: dict[str, str] = Field(..., description="Endpoints disponíveis")


# SCHEMAS DO ADK (/adk) - Formato conversacional

class ADKPredictRequest(BaseModel):
    """
    Schema simplificado para chamada via Google ADK.
    
    Parâmetros que um estudante forneceria em linguagem natural.
    Todos opcionais para máxima flexibilidade conversacional.
    """
    idade: int | None = Field(
        None,
        description="Idade do estudante em anos",
        ge=14,
        le=80
    )
    sexo: str | None = Field(
        None,
        description="Sexo (M ou F)"
    )
    raca: str | None = Field(
        None,

# TIPOS E DATACLASSES

        description="Raça/cor autodeclarada (BRANCA, PRETA, PARDA, AMARELA, INDIGENA)"
    )
    pcd: bool | None = Field(
        None,
        description="É pessoa com deficiência?"
    )
    uf: str | None = Field(
        None,
        description="Estado (sigla, ex: SP, RJ)"
    )
    municipio: str | None = Field(
        None,
        description="Cidade onde mora"
    )
    curso: str | None = Field(
        None,
        description="Curso desejado (ex: DIREITO, MEDICINA, ENGENHARIA)"
    )
    turno: str | None = Field(
        None,
        description="Turno preferido (MATUTINO, VESPERTINO, NOTURNO, INTEGRAL)"
    )
    modalidade: str | None = Field(
        None,
        description="Modalidade (PRESENCIAL ou EAD)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "idade": 20,
                "sexo": "F",
                "raca": "PARDA",
                "pcd": False,
                "uf": "SP",
                "municipio": "SAO PAULO",
                "curso": "DIREITO",
                "turno": "NOTURNO",
                "modalidade": "PRESENCIAL"
            }
        }


class ADKPredictResponse(BaseModel):
    """
    Resposta humanizada para o ADK.
    
    Formato otimizado para ser apresentado em linguagem natural pelo agente.
    Probabilidades em percentual (0-100) para facilitar comunicação.
    """
    tipo_bolsa: str = Field(
        ...,
        description="Tipo de bolsa mais provável (INTEGRAL ou PARCIAL)"
    )
    probabilidade_integral: float = Field(
        ...,
        description="Chance de bolsa integral (0 a 100%)",
        ge=0.0,
        le=100.0
    )
    probabilidade_parcial: float = Field(
        ...,
        description="Chance de bolsa parcial (0 a 100%)",
        ge=0.0,
        le=100.0
    )
    confianca: str = Field(
        ...,
        description="Nível de confiança da predição (ALTA, MÉDIA, BAIXA)"
    )
    mensagem: str = Field(
        ...,
        description="Mensagem explicativa em linguagem natural"
    )


class ADKHealthResponse(BaseModel):
    """Health check específico para ADK."""
    status: str = Field(..., description="Status do serviço")
    service: str = Field(..., description="Nome do serviço")
    available_tools: list[str] = Field(..., description="Tools disponíveis")