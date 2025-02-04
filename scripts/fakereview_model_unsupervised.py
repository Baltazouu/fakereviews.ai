import re
import string
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE

file_path = "./data/fake_reviews_final.csv"

df = pd.read_csv(file_path)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text.strip()

df["text_"] = df["text_"].astype(str).apply(clean_text)

vectorization = TfidfVectorizer(ngram_range=(1,2), max_features=5000, sublinear_tf=True)
X_vectors = vectorization.fit_transform(df["text_"])

model_dir = "./models/unsupervised"
os.makedirs(model_dir, exist_ok=True)

vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
joblib.dump(vectorization, vectorizer_path)
print(f"Vectorizer sauvegardé : {vectorizer_path}")

print("\n⏳ Exécution de K-Means...")
num_clusters = 2 
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df["kmeans_cluster"] = kmeans.fit_predict(X_vectors)


kmeans_path = os.path.join(model_dir, "kmeans_model.pkl")
joblib.dump(kmeans, kmeans_path)
print(f"Modèle K-Means sauvegardé : {kmeans_path}")

print("Clustering K-Means terminé !")
print(df[["text_", "kmeans_cluster"]].head(10))

print("\nExécution de Isolation Forest...")
iso_forest = IsolationForest(contamination=0.5, random_state=42)  # 50% supposés faux
df["anomaly"] = iso_forest.fit_predict(X_vectors)

isolation_path = os.path.join(model_dir, "isolation_forest.pkl")
joblib.dump(iso_forest, isolation_path)
print(f"Modèle Isolation Forest sauvegardé : {isolation_path}")

print("Détection d'anomalies terminée !")
print(df[["text_", "anomaly"]].head(10))  

print("\nVisualisation des données avec t-SNE...")
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X_vectors.toarray())

plt.figure(figsize=(10, 6))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=df["kmeans_cluster"], cmap="viridis", alpha=0.5)
plt.colorbar(label="Cluster K-Means")
plt.title("Visualisation des avis avec t-SNE")
plt.show()

print("Visualisation terminée !")
