import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Sistema Preditivo de Obesidade",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Sistema Preditivo de Nível de Obesidade")
st.markdown("**Aluno:** Vitor Santos  \n**RM:** 366038")
st.markdown("---")

@st.cache_resource
def load_model():
    return joblib.load("obesity_model.pkl")

model = load_model()

st.subheader("📋 Preencha os dados")

Gender = st.selectbox("Gênero", ["Male", "Female"])
Age = st.number_input("Idade", 14, 80)
Height = st.number_input("Altura (cm)", 100, 230)
Weight = st.number_input("Peso (kg)", 30.0, 200.0)

family_history = st.selectbox("Histórico familiar?", ["yes", "no"])
FAVC = st.selectbox("Alimentos calóricos frequentes?", ["yes", "no"])
SMOKE = st.selectbox("Fuma?", ["yes", "no"])
SCC = st.selectbox("Monitora calorias?", ["yes", "no"])

FCVC = st.slider("Consumo de vegetais", 1.0, 3.0, 2.0)
NCP = st.slider("Número de refeições", 1.0, 4.0, 3.0)
CH2O = st.slider("Consumo de água", 1.0, 3.0, 2.0)
FAF = st.slider("Atividade física", 0.0, 3.0, 1.0)
TUE = st.slider("Uso de tecnologia", 0.0, 2.0, 1.0)

CAEC = st.selectbox(
    "Come entre refeições?",
    ["no", "Sometimes", "Frequently", "Always"]
)

CALC = st.selectbox(
    "Consome álcool?",
    ["no", "Sometimes", "Frequently", "Always"]
)

MTRANS = st.selectbox(
    "Meio de transporte",
    ["Automobile", "Bike", "Motorbike",
     "Public_Transportation", "Walking"]
)

if st.button("🔮 Realizar Predição"):

    input_data = pd.DataFrame([{
        "Gender": Gender,
        "Age": Age,
        "Height": Height,
        "Weight": Weight,
        "family_history": family_history,
        "FAVC": FAVC,
        "SMOKE": SMOKE,
        "SCC": SCC,
        "FCVC": FCVC,
        "NCP": NCP,
        "CH2O": CH2O,
        "FAF": FAF,
        "TUE": TUE,
        "CAEC": CAEC,
        "CALC": CALC,
        "MTRANS": MTRANS
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data).max()

    imc = Weight / ((Height / 100) ** 2)

    st.success(f"🧠 Classificação Prevista: **{prediction}**")
    st.info(f"📊 Confiança do Modelo: **{probability:.2%}**")
    st.write(f"⚖ IMC: {imc:.2f}")

st.markdown("---")
st.caption("Projeto Acadêmico - FIAP - 2026")
