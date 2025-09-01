import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
import os

MODEL_PATH = "email_model.pkl" #define o nome do arquivo que será salvo o modelo treinado.



def train_model():

    data = {
        "email": [
            "Reunião de planejamento amanhã às 10h",
            "Promoção! Compre agora e ganhe desconto",
            "Relatório de vendas do último mês",
            "Ganhe prêmios incríveis clicando aqui",
            "Atualização do projeto em andamento",
            "Oferta especial: 50% de desconto em produtos",
            "Convite para palestra sobre inovação",
            "Spam: Clique aqui para ganhar dinheiro fácil",
            "Análise de dados trimestral",
            "Newsletter semanal com dicas de produtividade",
            "Reunião cancelada devido a imprevisto",
            "Anúncio: Novos produtos disponíveis",
            "Feedback sobre a apresentação de ontem",
            "Ganhe um iPhone grátis respondendo esta pesquisa",
            "Planejamento estratégico para o próximo ano",
            "Desconto imperdível em viagens",
            "Discussão sobre metas de equipe",
            "Oferta limitada: Inscreva-se agora",
            "Relatório financeiro mensal",
            "Prêmio surpresa para funcionários destacados",
            "Atualização de software necessária",
            "Promoção relâmpago: Compre 1 leve 2",
            "Reunião de follow-up do projeto",
            "Ganhe pontos extras no programa de fidelidade",
            "Análise de mercado e tendências",
            "Oferta especial para clientes VIP",
            "Convite para evento corporativo",
            "Spam: Melhore sua vida financeira",
            "Revisão de código e testes",
            "Desconto de 30% em todos os itens",
            "Discussão sobre melhorias no processo",
            "Ganhe um carro novo participando",
            "Relatório de progresso semanal",
            "Promoção: Frete grátis em compras acima de R$100",
            "Reunião para alinhamento de equipe",
            "Oferta: Assinatura mensal com desconto",
            "Análise de desempenho individual",
            "Ganhe dinheiro trabalhando em casa",
            "Planejamento de férias coletivas",
            "Desconto especial para estudantes",
            "Discussão sobre novos desafios",
            "Spam: Aumente seu salário rapidamente",
            "Relatório de auditoria interna",
            "Promoção: Produtos em liquidação",
            "Reunião de brainstorming",
            "Ganhe prêmios respondendo perguntas",
            "Análise de riscos e oportunidades",
            "Oferta: Pacote completo com economia",
            "Convite para workshop de capacitação",
            "Spam: Descubra segredos milionários"
        ],
        "label": [
            "produtivo", "improdutivo", "produtivo", "improdutivo",
            "produtivo", "improdutivo", "produtivo", "improdutivo",
            "produtivo", "produtivo", "produtivo", "improdutivo",
            "produtivo", "improdutivo", "produtivo", "improdutivo",
            "produtivo", "improdutivo", "produtivo", "produtivo",
            "produtivo", "improdutivo", "produtivo", "improdutivo",
            "produtivo", "improdutivo", "produtivo", "improdutivo",
            "produtivo", "improdutivo", "produtivo", "improdutivo",
            "produtivo", "improdutivo", "produtivo", "improdutivo",
            "produtivo", "improdutivo", "produtivo", "improdutivo",
            "produtivo", "improdutivo", "produtivo", "improdutivo",
            "produtivo", "improdutivo", "produtivo", "improdutivo",
            "produtivo", "improdutivo"
        ]
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
