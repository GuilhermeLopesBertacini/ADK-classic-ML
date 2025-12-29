"""
Google Gemini Agent Client - PROUNI Scholarship Advisor

Interactive CLI agent that uses Google Gemini with function calling
to connect students with the PROUNI ML prediction API.

Requirements:
    - GOOGLE_API_KEY environment variable set
    - Local PROUNI API running on localhost:8000
    - pip install google-genai requests python-dotenv

Usage:
    python examples/gemini_client.py
"""
import os
import sys

import requests
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_ENDPOINT = os.getenv("PROUNI_API_URL", "http://localhost:8000/adk/predict-bolsa")
GEMINI_MODEL = "gemini-2.0-flash-exp"


def consultar_sistema_prouni(
    idade: int,
    curso: str,
    uf: str,
    sexo: str,
    raca: str = "PARDA",
    turno: str = "NOTURNO",
    municipio: str | None = None,
    pcd: bool = False,
    modalidade: str = "PRESENCIAL"
) -> dict:
    """
    Consulta o sistema ML do PROUNI para prever tipo de bolsa.
    
    Args:
        idade: Idade do estudante (14-80)
        curso: Nome do curso desejado (ex: DIREITO, MEDICINA)
        uf: Estado (sigla, ex: SP, RJ)
        sexo: M ou F
        raca: BRANCA, PRETA, PARDA, AMARELA, INDIGENA
        turno: MATUTINO, VESPERTINO, NOTURNO, INTEGRAL
        municipio: Nome da cidade (opcional)
        pcd: Pessoa com deficiência
        modalidade: PRESENCIAL ou EAD
    
    Returns:
        dict com tipo_bolsa, probabilidades e mensagem
    """
    payload = {
        "idade": int(idade),
        "curso": curso.upper(),
        "uf": uf.upper(),
        "sexo": sexo.upper(),
        "raca": raca.upper(),from pathlib import Path

        "turno": turno.upper(),
        "pcd": pcd,
        "modalidade": modalidade.upper()
    }
    
    if municipio:
        payload["municipio"] = municipio.upper()
    
    print(f"\nConsultando API: {API_ENDPOINT}")
    print(f"Payload: {payload}")

    try:
        response = requests.post(API_ENDPOINT, json=payload, timeout=10)
        response.raise_for_status()
        resultado = response.json()
        
        print(f"✅ Resultado: {resultado['tipo_bolsa']} "
              f"({resultado['probabilidade_integral']:.1f}% integral)")
        
        return resultado
        
    except requests.RequestException as e:
        error_msg = f"Erro na conexão com API local: {str(e)}"
        print(f"{error_msg}")
        return {"erro": error_msg}


def setup_gemini_client():
    """Initialize Gemini client with API key validation."""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("Erro: GOOGLE_API_KEY não configurada")
        print("\nConfigure com:")
        print("  Crie um arquivo .env na raiz do projeto com:")
        print("  GOOGLE_API_KEY=sua-chave-aqui")
        sys.exit(1)
    
    client = genai.Client(api_key=api_key)
    return client


def run_interactive_session(client):
    """Run interactive CLI session."""
    print("\n" + "="*60)
    print("AGENTE PROUNI - Google Gemini + ML Predictions")
    print("="*60)
    print("\nExemplos de perguntas:")
    print("  • Tenho 20 anos, quero fazer Direito em SP. Tenho chance?")
    print("  • Sou mulher, parda, 22 anos, quero Medicina no RJ")
    print("  • Engenharia em MG, turno noturno, tenho 19 anos")
    print("\nDigite 'sair' para encerrar\n")

    # Define the tool for Gemini
    tool = {
        "function_declarations": [{
            "name": "consultar_sistema_prouni",
            "description": "Consulta o sistema ML do PROUNI para prever tipo de bolsa (INTEGRAL ou PARCIAL) com base no perfil do estudante",
            "parameters": {
                "type": "object",
                "properties": {
                    "idade": {
                        "type": "integer",
                        "description": "Idade do estudante (14-80 anos)"
                    },
                    "curso": {
                        "type": "string",
                        "description": "Nome do curso desejado (ex: DIREITO, MEDICINA, ENGENHARIA)"
                    },
                    "uf": {
                        "type": "string",
                        "description": "Estado brasileiro (sigla, ex: SP, RJ, MG)"
                    },
                    "sexo": {
                        "type": "string",
                        "description": "Sexo (M ou F)"
                    },
                    "raca": {
                        "type": "string",
                        "description": "Raça/cor autodeclarada: BRANCA, PRETA, PARDA, AMARELA, INDIGENA",
                        "default": "PARDA"
                    },
                    "turno": {
                        "type": "string",
                        "description": "Turno do curso: MATUTINO, VESPERTINO, NOTURNO, INTEGRAL",
                        "default": "NOTURNO"
                    },
                    "municipio": {
                        "type": "string",
                        "description": "Nome da cidade (opcional)"
                    },
                    "pcd": {
                        "type": "boolean",
                        "description": "É pessoa com deficiência?",
                        "default": False
                    },
                    "modalidade": {
                        "type": "string",
                        "description": "Modalidade de ensino: PRESENCIAL ou EAD",
                        "default": "PRESENCIAL"
                    }
                },
                "required": ["idade", "curso", "uf", "sexo"]
            }
        }]
    }

    # Start chat with tool
    chat = client.chats.create(
        model=GEMINI_MODEL,
        config={"tools": [tool]}
    )

    while True:
        try:
            user_msg = input("Você: ").strip()
            
            if not user_msg:
                continue
                
            if user_msg.lower() in ['sair', 'exit', 'quit']:
                print("\n👋 Até logo!")
                break
            
            # Send message
            response = chat.send_message(user_msg)
            
            # Check if there's a function call
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    # Execute the function
                    func_call = part.function_call
                    if func_call.name == "consultar_sistema_prouni":
                        result = consultar_sistema_prouni(**dict(func_call.args))
                        
                        # Send function result back
                        response = chat.send_message({
                            "function_response": {
                                "name": func_call.name,
                                "response": result
                            }
                        })
            
            # Get final text response
            final_text = ""
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text'):
                    final_text += part.text
            
            if final_text:
                print(f"\nAgente: {final_text}\n")
            
        except KeyboardInterrupt:
            print("\n\nSessão interrompida. Até logo!")
            break
        except Exception as e:
            print(f"\nErro: {e}\n")


def main():
    """Main entry point."""
    client = setup_gemini_client()
    run_interactive_session(client)


if __name__ == "__main__":
    main()
