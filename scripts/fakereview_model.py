import re
import string
import joblib
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier


file_path = "./data/fake_reviews_final.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")

df = pd.read_csv(file_path)

required_columns = {"label", "text_"}
if not required_columns.issubset(df.columns):
    raise KeyError(f"Les colonnes requises {required_columns} sont absentes du DataFrame.")

df_fake = df[df["label"] == "CG"].copy()
df_true = df[df["label"] == "OR"].copy()

df_fake["class"] = 0
df_true["class"] = 1

df_fake = df_fake.iloc[:-500]
df_true = df_true.iloc[:-500]

df_merge = pd.concat([df_fake, df_true], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

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

X = df_merge["text_"]
y = df_merge["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

vectorization = TfidfVectorizer(ngram_range=(1,2), max_features=5000, sublinear_tf=True)
Xv_train = vectorization.fit_transform(X_train)
Xv_test = vectorization.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_split=10),
    "Gradient Boosting": GradientBoostingClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=0),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

for name, model in models.items():
    print(f"\nEntraînement du modèle : {name}")
    model.fit(Xv_train, y_train)
    predictions = model.predict(Xv_test)
    score = model.score(Xv_test, y_test)
    print(f"Score {name} : {score:.4f}")
    print(classification_report(y_test, predictions))

model_dir = "./models"
os.makedirs(model_dir, exist_ok=True)

for name, model in models.items():
    model_filename = os.path.join(model_dir, f"{name.lower().replace(' ', '_')}.pkl")
    joblib.dump(model, model_filename)
    print(f"Modèle sauvegardé : {model_filename}")

vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
joblib.dump(vectorization, vectorizer_path)
print(f"Vectorizer sauvegardé : {vectorizer_path}")

