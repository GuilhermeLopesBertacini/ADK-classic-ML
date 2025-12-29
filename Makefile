.PHONY: help venv install inspect train serve test clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn
PYTEST := $(VENV)/bin/pytest

help:
	@echo "Targets disponíveis:"
	@echo "  make venv     - Cria ambiente virtual .venv (se não existir)"
	@echo "  make install  - Instala dependências em .venv"
	@echo "  make inspect  - Inspeciona o CSV do PROUNI"
	@echo "  make train    - Treina e salva o modelo (joblib)"
	@echo "  make serve    - Sobe a API FastAPI (uvicorn)"
	@echo "  make test     - Roda testes"
	@echo "  make clean    - Remove cache Python e arquivos temporários"

venv:
	@test -d $(VENV) || python3 -m venv $(VENV)
	@echo "Ambiente virtual criado em $(VENV)"

install: venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt
	@echo "Dependências instaladas com sucesso"

inspect: venv
	@echo "Inspecionando dados do PROUNI..."
	PYTHONPATH=src $(PYTHON) scripts/inspect_data.py

train: venv
	@echo "Treinando modelo..."
	@mkdir -p models
	PYTHONPATH=src $(PYTHON) -m prouni_agent.train \
		--data data/ProuniRelatorioDadosAbertos2020.csv \
		--out models/prouni_2020.joblib

serve: venv
	@echo "Iniciando servidor FastAPI em http://localhost:8000"
	@echo "Docs disponível em: http://localhost:8000/api/v1/docs"
	PYTHONPATH=src $(UVICORN) prouni_agent.api:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload

test: venv
	@echo "Executando testes..."
	PYTHONPATH=src $(PYTEST) -q

clean:
	@echo "Limpando arquivos temporários..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Limpeza concluída"
