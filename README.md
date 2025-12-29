# PROUNI Scholarship Prediction API

Sistema de Machine Learning para predição de tipo de bolsa PROUNI (INTEGRAL vs PARCIAL) baseado em dados históricos de 2020.

## Descrição

Este projeto treina um modelo de classificação usando scikit-learn para prever se um estudante tem maior probabilidade de receber bolsa integral ou parcial no PROUNI, com base em características demográficas e socioeconômicas.

O modelo é exposto via API REST (FastAPI) e inclui integração com Google Gemini para interface conversacional.

## Requisitos

- Python 3.13+
- pip
- virtualenv

## Instalação

```bash
git clone <repository-url>
cd be-solution

make install
```

Ou manualmente:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

### 1. Treinar o Modelo

```bash
make train
```

Isso irá:
- Ler o dataset de data/ProuniRelatorioDadosAbertos2020.csv
- Processar e transformar os dados
- Treinar um modelo LogisticRegression
- Salvar o pipeline completo em models/prouni_2020.joblib
- Exibir métricas de avaliação (precision, recall, ROC-AUC)

### 2. Iniciar a API

```bash
make serve
```

A API estará disponível em:
- http://localhost:8000
- Documentação interativa: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 3. Testar a API

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ano_concessao_bolsa": 2020,
    "sexo_beneficiario": "F",
    "raca_beneficiario": "PARDA",
    "data_nascimento": "2000-05-15",
    "beneficiario_deficiente_fisico": "N",
    "regiao_beneficiario": "SUDESTE",
    "uf_beneficiario": "SP",
    "municipio_beneficiario": "SAO PAULO",
    "modalidade_ensino_bolsa": "PRESENCIAL",
    "nome_curso_bolsa": "DIREITO",
    "nome_turno_curso_bolsa": "NOTURNO"
  }'
```

Resposta:
```json
{
  "label": "INTEGRAL",
  "proba_integral": 0.73,
  "proba_parcial": 0.27
}
```

### 4. Usar Cliente Gemini (Opcional)

Crie arquivo `.env` na raiz do projeto:

```
GOOGLE_API_KEY=sua-chave-aqui
```

Obtenha a chave em: https://aistudio.google.com/

Inicie a API (Terminal 1):
```bash
make serve
```

Execute o cliente (Terminal 2):
```bash
python examples/gemini_client.py
```

Interaja em linguagem natural:
```
Você: Tenho 20 anos, quero Direito em SP. Sou mulher, parda. Tenho chance?
Agente: Com base no seu perfil, você tem 73% de chance de bolsa integral...
```

## Endpoints da API

### POST /api/v1/predict
Predição técnica com campos estruturados.

### POST /adk/predict-bolsa
Predição conversacional com campos simplificados para integração ADK.

### GET /api/v1/health
Status da API e verificação de modelo carregado.

### GET /
Informações básicas e lista de endpoints.

## Estrutura do Projeto

```
be-solution/
├── data/                          # Datasets (CSV)
├── models/                        # Modelos treinados (.joblib)
├── src/prouni_agent/              # Código principal
│   ├── api.py                     # FastAPI application
│   ├── config.py                  # Configurações e paths
│   ├── data.py                    # Leitura e limpeza de dados
│   ├── features.py                # Engenharia de features
│   ├── modeling.py                # Pipeline scikit-learn
│   ├── predict.py                 # Funções de predição
│   ├── schemas.py                 # Pydantic models
│   ├── service.py                 # Serviço de predição
│   └── train.py                   # Script de treinamento
├── scripts/                       # Scripts auxiliares
│   └── inspect_data.py            # Inspeção do dataset
├── examples/                      # Exemplos de uso
│   └── gemini_client.py           # Cliente Google Gemini
├── tests/                         # Testes
├── Makefile                       # Comandos úteis
└── requirements.txt               # Dependências Python
```

## Comandos Make

```bash
make install    # Criar venv e instalar dependências
make train      # Treinar modelo
make serve      # Iniciar API
make test       # Executar testes
make clean      # Limpar cache e arquivos temporários
```

## Observações Importantes

O dataset do PROUNI 2020 contém apenas registros de bolsas concedidas. O modelo classifica INTEGRAL vs PARCIAL dado que houve concessão, não prevê probabilidade de conseguir bolsa vs não conseguir.

Features principais utilizadas:
- IDADE (calculada de DATA_NASCIMENTO)
- SEXO_BENEFICIARIO
- RACA_BENEFICIARIO
- BENEFICIARIO_DEFICIENTE_FISICO
- REGIAO_BENEFICIARIO
- UF_BENEFICIARIO
- MUNICIPIO_BENEFICIARIO
- MODALIDADE_ENSINO_BOLSA
- NOME_CURSO_BOLSA
- NOME_TURNO_CURSO_BOLSA

Métricas típicas (conjunto de teste):
- ROC-AUC: ~0.79
- Precision INTEGRAL: ~0.92
- Recall INTEGRAL: ~0.69
- Precision PARCIAL: ~0.41
- Recall PARCIAL: ~0.77

## Licença

MIT
