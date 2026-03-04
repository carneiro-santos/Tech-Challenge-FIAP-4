import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =============================
# Configuração
# =============================

st.set_page_config(
    page_title="Sistema Preditivo de Obesidade",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Sistema Preditivo de Nível de Obesidade")
st.markdown("Modelo baseado em Random Forest treinado com variáveis biométricas.")

# =============================
# Carregar modelo com segurança
# =============================

@st.cache_resource
def load_model():
    try:
        model = joblib.load("obesity_model.pkl")
        return model
    except Exception as e:
        st.error("Erro ao carregar modelo. Verifique versão do sklearn.")
        st.stop()

model = load_model()

# =============================
# Inputs organizados
# =============================

col1, col2 = st.columns(2)

with col1:
    idade = st.number_input("Idade", 10, 100, 25)
    altura = st.number_input("Altura (m)", 1.0, 2.5, 1.70)

with col2:
    peso = st.number_input("Peso (kg)", 30.0, 200.0, 70.0)
    atividade = st.slider("Frequência de Atividade Física (0-3)", 0, 3, 1)

# =============================
# Botão de predição
# =============================

if st.button("🔎 Realizar Predição"):

    if altura <= 0:
        st.warning("Altura inválida.")
        st.stop()

    imc = peso / (altura ** 2)

    dados = pd.DataFrame({
        "Age": [idade],
        "Height": [altura],
        "Weight": [peso],
        "FAF": [atividade],
        "BMI": [imc]
    })

    try:
        resultado = model.predict(dados)[0]

        st.success(f"🎯 Nível de Obesidade Predito: {resultado}")
        st.info(f"IMC Calculado: {imc:.2f}")

        st.markdown("---")
        st.markdown("### 📊 Interpretação")
        st.write("O resultado é baseado em padrões aprendidos pelo modelo Random Forest.")

    except Exception as e:
        st.error("Erro durante a predição.")
        st.exception(e)
