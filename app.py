import streamlit as st
from model import predict_email

st.title("📧 Classificador de Emails")
st.write("Classifique emails como **produtivos** ou **improdutivos**")

email_input = st.text_area("Cole o conteúdo do email aqui:")

if st.button("Classificar"):
    if email_input.strip():
        result = predict_email(email_input)
        st.success(f"O email foi classificado como: **{result.upper()}**")
    else:
        st.warning("Por favor, insira o conteúdo do email.")
