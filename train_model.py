import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ===============================
# 1. Carregar base de dados
# ===============================

df = pd.read_csv("Obesity.csv")

# ===============================
# 2. Engenharia de Features
# ===============================

# Criar IMC
df["BMI"] = df["Weight"] / (df["Height"] ** 2)

# Selecionar apenas variáveis numéricas
features = ["Age", "Height", "Weight", "FAF", "BMI"]
target = "NObeyesdad"

X = df[features]
y = df[target]

# ===============================
# 3. Dividir treino e teste
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 4. Criar Modelo
# ===============================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42
)

# ===============================
# 5. Treinar Modelo
# ===============================

model.fit(X_train, y_train)

# ===============================
# 6. Avaliação
# ===============================

y_pred = model.predict(X_test)

print("\n📊 RESULTADOS DO MODELO\n")
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 7. Importância das Variáveis
# ===============================

importances = pd.DataFrame({
    "Variável": features,
    "Importância": model.feature_importances_
}).sort_values(by="Importância", ascending=False)

print("\n🔎 Importância das Variáveis:\n")
print(importances)

# ===============================
# 8. Salvar Modelo
# ===============================

joblib.dump(model, "obesity_model.pkl")

print("\n✅ Modelo salvo como 'obesity_model.pkl'")
