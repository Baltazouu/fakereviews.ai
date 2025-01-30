import joblib
import pandas as pd
import re
import string

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

def output_label(n):
    return "Fake Review" if n == 0 else "Not A Fake Review"

def manual_testing(news):
    testing_news = {"text_": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text_"] = new_def_test["text_"].apply(wordopt)
    new_x_test = new_def_test["text_"]
    new_xv_test = vectorization.transform(new_x_test)
    
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    print(f"\n\nLR Prediction: {output_label(pred_LR[0])} \nDT Prediction: {output_label(pred_DT[0])} \nGBC Prediction: {output_label(pred_GBC[0])} \nRFC Prediction: {output_label(pred_RFC[0])}")

# Load saved models and vectorizer
LR = joblib.load('models/lr_model.pkl')
DT = joblib.load('models/dt_model.pkl')
GBC = joblib.load('models/gbc_model.pkl')
RFC = joblib.load('models/rfc_model.pkl')
vectorization = joblib.load('models/vectorizer.pkl')

# Manual testing
fake_review1 = input("Enter a fake or real review: ")
manual_testing(fake_review1)

fake_review2 = input("Enter a fake or real review: ")
manual_testing(fake_review2)