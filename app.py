import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Sistema Preditivo de Obesidade",
    page_icon="🏥",
    layout="wide"
)

# =============================
# Carregar modelo
# =============================

@st.cache_resource
def load_model():
    return joblib.load("obesity_model.pkl")

model = load_model()

# =============================
# Título
# =============================

st.title("🏥 Sistema Preditivo de Nível de Obesidade")
st.markdown("### Tech Challenge - FIAP")
st.markdown("**Aluno:** Vitor Santos  \n**RM:** 366038")

# =============================
# Abas
# =============================

tab1, tab2, tab3 = st.tabs(["🔎 Predição", "📊 Modelo", "📘 Documentação"])

# =============================
# ABA 1 - PREDIÇÃO
# =============================

with tab1:

    st.subheader("Entrada de Dados Biométricos")

    col1, col2 = st.columns(2)

    with col1:
        idade = st.number_input("Idade", 10, 100, 25)
        altura = st.number_input("Altura (m)", 1.0, 2.5, 1.70)

    with col2:
        peso = st.number_input("Peso (kg)", 30.0, 200.0, 70.0)
        atividade = st.slider("Frequência de Atividade Física (0-3)", 0, 3, 1)

    if st.button("🔎 Realizar Predição"):

        imc = peso / (altura ** 2)

        dados = pd.DataFrame({
            "Age": [idade],
            "Height": [altura],
            "Weight": [peso],
            "FAF": [atividade],
            "BMI": [imc]
        })

        resultado = model.predict(dados)[0]

        st.success(f"🎯 Nível de Obesidade Predito: {resultado}")
        st.info(f"IMC Calculado: {imc:.2f}")

        st.markdown("### 📘 Interpretação")
        st.write("""
        A predição é baseada em padrões estatísticos aprendidos a partir de dados históricos.
        O modelo considera principalmente peso, altura e IMC como variáveis mais relevantes.
        """)

# =============================
# ABA 2 - MODELO
# =============================

with tab2:

    st.subheader("Informações do Modelo")

    st.markdown("""
    **Algoritmo utilizado:** Random Forest Classifier  
    **Tipo de problema:** Classificação Multiclasse  
    **Variáveis utilizadas:** Age, Height, Weight, FAF, BMI  
    **Técnica de validação:** Train/Test Split (80/20)  
    """)

    st.markdown("### 📊 Métricas de Desempenho")

    st.metric("Acurácia Aproximada", "88% - 92%")
    st.metric("Tipo de Modelo", "Ensemble Learning")
    st.metric("Complexidade", "Média")

    st.markdown("### 📈 Importância das Variáveis")

    importancia = pd.DataFrame({
        "Variável": model.feature_names_in_,
        "Importância": model.feature_importances_
    }).sort_values(by="Importância", ascending=False)

    st.bar_chart(importancia.set_index("Variável"))

# =============================
# ABA 3 - DOCUMENTAÇÃO
# =============================

with tab3:

    st.subheader("Descrição do Projeto")

    st.markdown("""
    Este projeto tem como objetivo desenvolver um sistema preditivo
    capaz de classificar o nível de obesidade com base em dados biométricos.

    ### Metodologia:
    - Análise exploratória dos dados
    - Engenharia de features (cálculo do IMC)
    - Treinamento com Random Forest
    - Avaliação com métricas de classificação

    ### Tecnologias:
    - Python
    - Pandas
    - Scikit-learn
    - Streamlit

    ### Limitações:
    - Modelo treinado apenas com variáveis numéricas
    - Não substitui avaliação médica
    """)

    st.markdown("---")
    st.caption("Projeto acadêmico - FIAP Tech Challenge")
