import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Carregar base
df = pd.read_csv("Obesity.csv")

# Criar IMC
df["BMI"] = df["Weight"] / (df["Height"] ** 2)

# Selecionar apenas variáveis numéricas simplificadas
X = df[["Age", "Height", "Weight", "FAF", "BMI"]]
y = df["NObeyesdad"]

# Separar treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Criar modelo
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Treinar
model.fit(X_train, y_train)

# Avaliar
y_pred = model.predict(X_test)

print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Salvar modelo
joblib.dump(model, "obesity_model.pkl")

print("Modelo salvo com sucesso!")
