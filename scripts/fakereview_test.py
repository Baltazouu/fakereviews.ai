import joblib
import pandas as pd
import re
import string

def wordopt(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)  # Suppression des crochets et leur contenu
    text = re.sub(r"\W", " ", text)  # Suppression des caractères non alphanumériques
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Suppression des URLs
    text = re.sub(r"<.*?>+", "", text)  # Suppression des balises HTML
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)  # Suppression de la ponctuation
    text = re.sub(r"\n", " ", text)  # Suppression des sauts de ligne
    text = re.sub(r"\w*\d\w*", "", text)  # Suppression des mots contenant des chiffres
    return text.strip()

def output_label(n):
    return "Fake Review" if n == 0 else "Not A Fake Review"

try:
    LR = joblib.load("./models/logistic_regression.pkl")
    DT = joblib.load("./models/decision_tree.pkl")
    GBC = joblib.load("./models/gradient_boosting.pkl")
    RFC = joblib.load("./models/random_forest.pkl")
    XGB = joblib.load("./models/xgboost.pkl")
    vectorization = joblib.load("./models/vectorizer.pkl")
except Exception as e:
    print(f"Erreur lors du chargement des modèles : {e}")
    exit(1)

def manual_testing(review):
    try:
        test_df = pd.DataFrame({"text_": [review]})
        test_df["text_"] = test_df["text_"].apply(wordopt)
        
        new_xv_test = vectorization.transform(test_df["text_"])

        if new_xv_test.shape[1] != vectorization.get_feature_names_out().shape[0]:
            print("Erreur : Le vectorizer ne correspond pas au modèle entraîné.")
            return

        # Prédictions 
        pred_LR = LR.predict(new_xv_test)
        pred_DT = DT.predict(new_xv_test)
        pred_GBC = GBC.predict(new_xv_test)
        pred_RFC = RFC.predict(new_xv_test)
        pred_XGB = XGB.predict(new_xv_test)

        print(f"\nLR Prediction: {output_label(pred_LR[0])}")
        print(f"DT Prediction: {output_label(pred_DT[0])}")
        print(f"GBC Prediction: {output_label(pred_GBC[0])}")
        print(f"RFC Prediction: {output_label(pred_RFC[0])}")
        print(f"XGB Prediction: {output_label(pred_XGB[0])}\n")

    except Exception as e:
        print(f"Erreur lors de la prédiction : {e}")

# Tester manuellement
while True:
    user_input = input("Enter a review (or type 'exit' to quit): ").strip()
    if user_input.lower() == "exit":
        print("Exiting...")
        break
    manual_testing(user_input)



