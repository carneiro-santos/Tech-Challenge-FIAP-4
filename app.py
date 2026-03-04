import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Predição de Obesidade", layout="wide")

st.title("🏥 Sistema Preditivo de Nível de Obesidade")
st.write("Preencha os dados abaixo para realizar a predição.")

# Carregar modelo
model = joblib.load("obesity_model.pkl")

# Inputs
idade = st.number_input("Idade", min_value=10, max_value=100, value=25)
altura = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70)
peso = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0)
atividade = st.slider("Frequência de Atividade Física (0-3)", 0, 3, 1)

if st.button("Realizar Predição"):

    imc = peso / (altura ** 2)

    dados = pd.DataFrame({
        "Age": [idade],
        "Height": [altura],
        "Weight": [peso],
        "FAF": [atividade],
        "BMI": [imc]
    })

    resultado = model.predict(dados)[0]

    st.success(f"Nível de Obesidade Predito: {resultado}")
    st.info(f"IMC Calculado: {imc:.2f}")
