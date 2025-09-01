import requests
import os
from dotenv import load_dotenv
load_dotenv()
# Use variável de ambiente para o token (mais seguro)
HF_TOKEN = os.getenv("HF_TOKEN")  # Configure esta variável de ambiente
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

# Se não houver token configurado, informar ao usuário
if not HF_TOKEN:
    print("⚠️  ATENÇÃO: Configure a variável de ambiente HF_TOKEN com seu token do Hugging Face")
    print("   Exemplo: export HF_TOKEN='seu_token_aqui' (Linux/Mac)")
    print("   Ou: set HF_TOKEN=seu_token_aqui (Windows)")

headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def classify_email(email_text: str):
    if not HF_TOKEN:
        return "erro", 0.0

    payload = {
        "inputs": email_text,
        "parameters": {"candidate_labels": ["produtivo", "improdutivo"]}
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    data = response.json()

    if "labels" in data:
        label = data["labels"][0]
        score = data["scores"][0]
        return label, score
    else:
        return "erro", 0.0

if __name__ == "__main__":
    email = "As melhores praticas da produtividade, adquira ja o livro"
    label, score = classify_email(email)
    print(f"Classificação: {label} (confiança: {score:.2f})")
