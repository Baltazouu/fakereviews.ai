import re
import string
import joblib
import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Vérifier l'existence du fichier
file_path = "./data/fake_reviews_final.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")

# Chargement des données
df = pd.read_csv(file_path)

# Vérifier la présence des colonnes attendues
required_columns = {"label", "text_"}
if not required_columns.issubset(df.columns):
    raise KeyError(f"Les colonnes requises {required_columns} sont absentes du DataFrame.")

df_fake = df[df["label"] == "CG"].copy()
df_true = df[df["label"] == "OR"].copy()

# Assigner les labels de classe
df_fake["class"] = 0
df_true["class"] = 1

# Suppression des 500 dernières lignes
df_fake = df_fake.iloc[:-500]
df_true = df_true.iloc[:-500]

# Suppression des 10 dernières lignes pour les tests manuels
df_fake_manual_testing = df_fake.tail(10).copy()
df_true_manual_testing = df_true.tail(10).copy()

df_fake_manual_testing["class"] = 0
df_true_manual_testing["class"] = 1

# Fusion des jeux de données
df_merge = pd.concat([df_fake, df_true], axis=0).sample(frac=1).reset_index(drop=True)

# Vérification des valeurs manquantes
if df_merge.isnull().sum().any():
    df_merge = df_merge.dropna()

# Fonction de nettoyage du texte
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

df_merge["text_"] = df_merge["text_"].astype(str).apply(clean_text)

# Séparation des données en features et labels
X = df_merge["text_"]
y = df_merge["class"]

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Vectorisation avec n-grams
vectorization = TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_df=0.95, min_df=2)
Xv_train = vectorization.fit_transform(X_train)
Xv_test = vectorization.transform(X_test)

# Grille d'hyperparamètres pour les modèles
param_grid = {
    "Random Forest": {
        'n_estimators': [50, 100],  # Réduit pour l'optimisation de la vitesse
        'max_depth': [None, 10, 20]
    },
    "Gradient Boosting": {
        'n_estimators': [50],
        'learning_rate': [0.1],
        'max_depth': [5]
    }
}

# Modèles à utiliser
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# Grid Search et optimisation
for name, model in models.items():
    print(f"\nOptimisation du modèle : {name}")
    grid_search = GridSearchCV(model, param_grid[name], cv=3, scoring='accuracy', n_jobs=-1)  # n_jobs=-1 pour l'exécution parallèle
    grid_search.fit(Xv_train, y_train)
    print(f"Meilleurs hyperparamètres pour {name}: {grid_search.best_params_}")
    print(f"Meilleur score pour {name}: {grid_search.best_score_:.4f}")
    
    # Entraîner le modèle avec les meilleurs paramètres
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(Xv_test)
    score = best_model.score(Xv_test, y_test)
    print(f"Score final {name} : {score:.4f}")
    print(classification_report(y_test, predictions))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, predictions)
    print("Matrice de confusion :\n", cm)

    # Faux Positifs, Faux Négatifs, Vrais Positifs, Vrais Négatifs
    tn, fp, fn, tp = cm.ravel()
    print(f"\nFaux Positifs (FP) : {fp}")
    print(f"Faux Négatifs (FN) : {fn}")
    print(f"Vrais Positifs (TP) : {tp}")
    print(f"Vrais Négatifs (TN) : {tn}")

    # Analyser les faux positifs et faux négatifs
    faux_positifs = X_test[(predictions == 1) & (y_test == 0)]
    faux_negatifs = X_test[(predictions == 0) & (y_test == 1)]

    print(f"\nExemples de Faux Positifs :\n{faux_positifs.head()}")
    print(f"\nExemples de Faux Négatifs :\n{faux_negatifs.head()}")

# Sauvegarde des modèles
model_dir = "models2"
os.makedirs(model_dir, exist_ok=True)

for name, model in models.items():
    model_filename = os.path.join(model_dir, f"{name.lower().replace(' ', '_')}.pkl")
    joblib.dump(model, model_filename)
    print(f"Modèle sauvegardé : {model_filename}")

# Sauvegarde du vectorizer
vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
joblib.dump(vectorization, vectorizer_path)
print(f"Vectorizer sauvegardé : {vectorizer_path}")
