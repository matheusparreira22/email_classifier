import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
import os

MODEL_PATH = "email_model.pkl" #define o nome do arquivo que será salvo o modelo treinado.



def train_model():
    # Exemplo simples de dataset (pode ser substituído depois)
    data = {
        "email": [
            "Reunião de planejamento amanhã às 10h",
            "Promoção! Compre agora e ganhe desconto",
            "Relatório de vendas do último mês",
            "Ganhe prêmios incríveis clicando aqui"
        ],
        "label": ["produtivo", "improdutivo", "produtivo", "improdutivo"]
    }
    df = pd.DataFrame(data)#define o dataset

    #cria um vetor numerico
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(df["email"], df["label"])
    #salva o modelo treinado
    joblib.dump(model, MODEL_PATH)
    print(f"Modelo treinado e salvo em {MODEL_PATH}")

def predict_email(email_text: str):
    #carrega o arquivo do modelo treinado e verifica se ele existe
    if not os.path.exists(MODEL_PATH):
        #se nao existir treina o modelo
        train_model()
        #se existir carrega o modelo
    model = joblib.load(MODEL_PATH)
    #faz a predição
    prediction = model.predict([email_text])[0]
    return prediction


if __name__ == "__main__":
    train_model()
