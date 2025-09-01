import streamlit as st
import time
from model import predict_email
from hf_api import classify_email
from dotenv import load_dotenv
import os

load_dotenv()  # Carrega o .env
HF_TOKEN = os.getenv('HF_TOKEN')
st.title("📧 Classificador de Emails")
st.write("Classifique emails como **produtivos** ou **improdutivos**")

# Opção para escolher o método de classificação
method = st.selectbox(
    "Escolha o método de classificação:",
    ("Modelo Treinado (Local)", "Hugging Face API")
)

email_input = st.text_area("Cole o conteúdo do email aqui:")

if st.button("Classificar"):
    if email_input.strip():
        start_time = time.time()

        if method == "Modelo Treinado (Local)":
            result = predict_email(email_input)
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.success(f"O email foi classificado como: **{result.upper()}**")
            st.info(f"⏱️ Tempo de resposta: {elapsed_time:.3f} segundos")

        elif method == "Hugging Face API":
            label, score = classify_email(email_input)
            end_time = time.time()
            elapsed_time = end_time - start_time

            if label != "erro":
                st.success(f"O email foi classificado como: **{label.upper()}** (Confiança: {score:.2f})")
                st.info(f"⏱️ Tempo de resposta: {elapsed_time:.3f} segundos")
            else:
                st.error("Erro ao classificar o email. Tente novamente.")
                st.info(f"⏱️ Tempo até erro: {elapsed_time:.3f} segundos")
    else:
        st.warning("Por favor, insira o conteúdo do email.")
