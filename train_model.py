import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ===============================
# 1. Carregar Dataset
# ===============================

df = pd.read_csv("Obesity.csv")

# ===============================
# 2. Engenharia de Feature
# ===============================

# Criar IMC
df["BMI"] = df["Weight"] / (df["Height"] ** 2)

# Selecionar apenas variáveis numéricas
features = ["Age", "Height", "Weight", "FAF", "BMI"]
target = "NObeyesdad"

X = df[features]
y = df[target]

# ===============================
# 3. Dividir Treino/Teste
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
    random_state=42
)

# ===============================
# 5. Treinar
# ===============================

model.fit(X_train, y_train)

# ===============================
# 6. Avaliar
# ===============================

y_pred = model.predict(X_test)

print("\n📊 RESULTADOS DO MODELO\n")
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 7. Salvar Modelo
# ===============================

joblib.dump(model, "obesity_model.pkl")

print("\n✅ Modelo salvo com sucesso como 'obesity_model.pkl'")
