import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# Charger les modèles spécifiques
model_dir = "./models"
models_filenames = [
    "decision_tree.pkl",
    "gradient_boosting.pkl",
    "logistic_regression.pkl",
    "random_forest.pkl"
]

vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
if not os.path.exists(vectorizer_path):
    raise FileNotFoundError("Vectoriseur introuvable. Assurez-vous qu'il est bien sauvegardé.")

vectorizer = joblib.load(vectorizer_path)

# Charger les modèles disponibles
models = {}
for model_file in models_filenames:
    model_path = os.path.join(model_dir, model_file)
    if os.path.exists(model_path):
        model_name = model_file.replace(".pkl", "").replace("_", " ").title()
        models[model_name] = joblib.load(model_path)
    else:
        print(f"Modèle introuvable : {model_file}")

# Charger les données
file_path = "./data/fake_reviews_final.csv"
df = pd.read_csv(file_path)

# Vérifier la présence des colonnes attendues
if "label" not in df.columns or "text_" not in df.columns:
    raise KeyError("Les colonnes requises sont absentes du DataFrame.")

# Sélection des 500 dernières lignes
df_test = df.tail(500).copy()
df_test["class"] = df_test["label"].apply(lambda x: 0 if x == "CG" else 1)

# Prétraitement du texte
def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    return text.strip()

df_test["text_"] = df_test["text_"].astype(str).apply(clean_text)

# Transformation avec TF-IDF
X_test = vectorizer.transform(df_test["text_"])
y_test = df_test["class"]

# Tester tous les modèles
results = {}
confusion_matrices = {}
classification_reports = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    correct = sum(y_pred == y_test)
    incorrect = sum(y_pred != y_test)
    results[name] = {"correct": correct, "incorrect": incorrect, "accuracy": accuracy}
    
    # Matrice de confusion
    confusion_matrices[name] = confusion_matrix(y_test, y_pred)
    classification_reports[name] = classification_report(y_test, y_pred, output_dict=True)

# Convertir en DataFrame
df_results = pd.DataFrame(results).transpose()

# Afficher les résultats
plt.figure(figsize=(12, 6))
df_results[["correct", "incorrect"]].plot(kind="bar", stacked=True, figsize=(12, 6))
plt.title("Comparaison des Modèles sur les 500 Derniers Avis")
plt.xlabel("Modèles")
plt.ylabel("Nombre d'Avis")
plt.xticks(rotation=45)
plt.legend(["Corrects", "Incorrects"])
plt.show()

# Matrices de confusion
for name, cm in confusion_matrices.items():
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Faux", "Vrai"], yticklabels=["Faux", "Vrai"])
    plt.xlabel("Prédiction")
    plt.ylabel("Réel")
    plt.title(f"Matrice de Confusion - {name}")
    plt.show()

# Rapports de classification
for name, report in classification_reports.items():
    print(f"\nRapport de classification pour {name} :")
    print(pd.DataFrame(report).transpose())