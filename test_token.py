#!/usr/bin/env python3
"""
Script de teste para verificar se o token do Hugging Face está funcionando
Execute: python test_token.py
"""

from dotenv import load_dotenv
import os
import requests

def test_token():
    # Carrega o .env
    load_dotenv()

    # Obtém o token
    token = os.getenv('HF_TOKEN')

    if not token:
        print("❌ ERRO: Token não encontrado no arquivo .env")
        print("💡 Verifique se o arquivo .env existe e contém HF_TOKEN=seu_token")
        return False

    print(f"✅ Token encontrado: {token[:10]}...")

    # Testa a API
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": "Este é um teste de email produtivo sobre trabalho",
        "parameters": {"candidate_labels": ["produtivo", "improdutivo"]}
    }

    try:
        print("🔄 Testando conexão com Hugging Face API...")
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)

        print(f"📊 Status da resposta: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            if "labels" in data:
                label = data["labels"][0]
                score = data["scores"][0]
                print("✅ API funcionando perfeitamente!")
                print(f"📧 Classificação de teste: {label.upper()} (Confiança: {score:.2f})")
                return True
            else:
                print(f"❌ Resposta inesperada da API: {data}")
                return False

        elif response.status_code == 403:
            print("❌ ERRO 403: Token sem permissões suficientes")
            print("💡 Soluções:")
            print("   1. Vá para https://huggingface.co/settings/tokens")
            print("   2. Crie um novo token com permissões 'Read'")
            print("   3. Ou aceite os termos do modelo em https://huggingface.co/facebook/bart-large-mnli")
            return False

        else:
            print(f"❌ Erro na API ({response.status_code}): {response.text[:100]}")
            return False

    except requests.exceptions.Timeout:
        print("❌ Timeout: A API demorou muito para responder")
        return False
    except Exception as e:
        print(f"❌ Erro de conexão: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Teste do Token Hugging Face")
    print("=" * 40)

    success = test_token()

    if success:
        print("\n🎉 Tudo funcionando! Você pode usar o Hugging Face no app.")
    else:
        print("\n⚠️  Problemas detectados. Verifique as soluções acima.")

    print("\n💡 Dica: Execute 'streamlit run app.py' para testar no aplicativo completo")
