import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Predição de Obesidade", layout="wide")

st.title("🏥 Sistema Preditivo de Nível de Obesidade")
st.write("Preencha os dados abaixo para realizar a predição.")

# Carregar modelo
model = joblib.load("obesity_model.pkl")

# -----------------------------
# Inputs
# -----------------------------

idade = st.number_input("Idade", 10, 100, 25)
altura = st.number_input("Altura (m)", 1.0, 2.5, 1.70)
peso = st.number_input("Peso (kg)", 30.0, 200.0, 70.0)

genero = st.selectbox("Gênero", ["Male", "Female"])

atividade = st.slider("Frequência de Atividade Física (0-3)", 0, 3, 1)
tempo_tela = st.slider("Tempo de Tela (0-3)", 0, 3, 1)
agua = st.slider("Consumo de Água (0-3)", 0, 3, 2)
vegetais = st.slider("Consumo de Vegetais (0-3)", 0, 3, 2)
refeicoes = st.slider("Número de Refeições (1-4)", 1, 4, 3)

historico = st.selectbox("Histórico Familiar de Obesidade", ["yes", "no"])
alto_calorico = st.selectbox("Consumo Frequente de Alta Caloria", ["yes", "no"])
fumante = st.selectbox("Fumante", ["yes", "no"])
monitora = st.selectbox("Monitora Calorias", ["yes", "no"])

caec = st.selectbox("Consumo Entre Refeições", ["no", "Sometimes", "Frequently", "Always"])
alcool = st.selectbox("Consumo de Álcool", ["no", "Sometimes", "Frequently", "Always"])

transporte = st.selectbox(
    "Meio de Transporte",
    ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"]
)

# -----------------------------
# Predição
# -----------------------------

if st.button("Realizar Predição"):

    imc = peso / (altura ** 2)

    dados = pd.DataFrame({
        "Age": [idade],
        "Height": [altura],
        "Weight": [peso],
        "Gender": [genero],
        "FCVC": [vegetais],
        "NCP": [refeicoes],
        "CH2O": [agua],
        "FAF": [atividade],
        "TUE": [tempo_tela],
        "family_history": [historico],
        "FAVC": [alto_calorico],
        "SMOKE": [fumante],
        "SCC": [monitora],
        "CAEC": [caec],
        "CALC": [alcool],
        "MTRANS": [transporte],
        "BMI": [imc]
    })

    resultado = model.predict(dados)[0]

    st.success(f"Nível de Obesidade Predito: {resultado}")
    st.info(f"IMC Calculado: {imc:.2f}")
