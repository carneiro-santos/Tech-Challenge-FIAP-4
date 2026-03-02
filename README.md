# 🏥 Tech Challenge – Fase 4  
## Modelo Preditivo para Classificação do Nível de Obesidade

**Aluno:** Vitor Santos  
**Curso:** Pós Tech – Data Analytics  
**Ano:** 2026  

---

## 📌 1. Contexto do Problema

A obesidade é uma condição médica caracterizada pelo acúmulo excessivo de gordura corporal, sendo considerada um dos principais problemas de saúde pública mundial.

Este projeto tem como objetivo desenvolver um modelo de Machine Learning capaz de auxiliar profissionais da saúde na identificação do nível de obesidade de um paciente, com base em características físicas e comportamentais.

---

## 🎯 2. Objetivo

Desenvolver um sistema preditivo capaz de classificar indivíduos em diferentes níveis de obesidade, utilizando:

- Dados físicos
- Hábitos alimentares
- Estilo de vida
- Indicadores comportamentais

Além disso, construir um painel analítico para apoiar a tomada de decisão clínica.

---

## 📊 3. Base de Dados

O dataset contém variáveis relacionadas a:

- Idade
- Altura
- Peso
- Frequência de atividade física
- Consumo de água
- Histórico familiar
- Hábitos alimentares
- Uso de dispositivos eletrônicos

Foi criada a variável **IMC (Índice de Massa Corporal)**:

IMC = Peso (kg) / Altura² (m)

---

## 🧠 4. Metodologia

### 🔹 Tratamento dos Dados
- Conversão de variáveis binárias
- Mapeamento de variáveis ordinais
- Tradução completa para português
- Criação da variável IMC

### 🔹 Modelagem
- Divisão treino/teste (80/20)
- Pipeline com `ColumnTransformer`
- Modelo Random Forest
- Avaliação com Accuracy, F1-score e Matriz de Confusão

---

## 📈 5. Resultados

O modelo apresentou:

- 🎯 **Acurácia: 98,8%**
- Alto desempenho em todas as classes
- Coerência clínica validada pela análise de importância das variáveis

A variável IMC demonstrou forte impacto preditivo.

---

## 📊 6. Dashboard Analítico

Painel desenvolvido no Looker Studio para análise exploratória e insights clínicos:

🔗 **Acesse o Dashboard:**  
https://lookerstudio.google.com/s/gWGZBXIe7eI

O painel apresenta:

- Distribuição dos níveis de obesidade
- IMC médio por classe
- Relação entre atividade física e obesidade
- Influência do histórico familiar
- Indicadores comportamentais

---

## 🚀 7. Aplicação Preditiva (Streamlit)

Sistema interativo para predição em tempo real:

🔗 **Acesse a Aplicação:**  
(https://tech-challenge-fiap-4-rm366038.streamlit.app)

Funcionalidades:

- Inserção manual dos dados
- Cálculo automático do IMC
- Classificação do nível de obesidade
- Probabilidade da predição
- Interpretação clínica

---

## 🗂 8. Estrutura do Projeto


Tech-Challenge-FIAP-4/
│
├── data/
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_modelagem.ipynb
│
├── ml/
│ └── obesity_model.pkl
│
├── streamlit/
│ ├── app.py
│ └── requirements.txt
│
└── README.md



---

## 🎓 9. Conclusão

O modelo desenvolvido demonstrou alta capacidade preditiva e coerência clínica, podendo ser utilizado como ferramenta de apoio à decisão médica.

A integração entre Machine Learning e Business Intelligence permitiu transformar dados brutos em insights estratégicos.

---
