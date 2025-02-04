import os
import joblib
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

model_dir = "./models/unsupervised"
vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
kmeans_path = os.path.join(model_dir, "kmeans_model.pkl")
isolation_path = os.path.join(model_dir, "isolation_forest.pkl")


vectorizer = joblib.load(vectorizer_path)
kmeans = joblib.load(kmeans_path)
iso_forest = joblib.load(isolation_path)

file_path = "./data/fake_reviews_final.csv"
df = pd.read_csv(file_path)

# Prétraiter les textes
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    return text.strip()

df["text_"] = df["text_"].astype(str).apply(clean_text)

X = vectorizer.transform(df["text_"])

print("\nEvaluation de K-Means...")
kmeans_pred = kmeans.predict(X)
silhouette_kmeans = silhouette_score(X, kmeans_pred)  # Mesure de cohésion du clustering

print("\n⏳ Evaluation de Isolation Forest...")
iso_forest_pred = iso_forest.predict(X)
iso_forest_pred = (iso_forest_pred == 1).astype(int)

df["true_labels"] = df["label"].apply(lambda x: 0 if x == "CG" else 1)

correct_kmeans = sum(kmeans_pred == df["true_labels"])
correct_iso_forest = sum(iso_forest_pred == df["true_labels"])

results = {
    "K-Means": {
        "correct": correct_kmeans,
        "incorrect": len(df) - correct_kmeans,
        "silhouette_score": silhouette_kmeans
    },
    "Isolation Forest": {
        "correct": correct_iso_forest,
        "incorrect": len(df) - correct_iso_forest,
        "silhouette_score": None  
    }
}

df_results = pd.DataFrame(results).transpose()

plt.figure(figsize=(12, 6))

df_results[["correct", "incorrect"]].plot(kind="bar", stacked=True, figsize=(12, 6))
plt.title("Comparaison des Modèles Non Supervisés sur les Avis")
plt.xlabel("Modèles")
plt.ylabel("Nombre d'Avis")
plt.xticks(rotation=45)
plt.legend(["Corrects", "Incorrects"])

for i, value in enumerate(df_results["silhouette_score"].dropna()):
    plt.text(i, df_results["correct"][i] + 50, f"Silhouette: {value:.2f}", ha="center", fontsize=12, color="black")

plt.show()
