import re
import string
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier




# Load datas
df = pd.read_csv("./data/fake_reviews.csv")

# Filter by type
df_fake = df[df["label"] == "CG"]  # Fake reviews (Computer Generated)
df_true = df[df["label"] == "OR"]  # True reviews (Original Reviews)


df_fake.head()
df_true.head(5)

df_fake["class"] = 0
df_true["class"] = 1

df_fake.shape, df_true.shape


# Removing last 10 rows for manual testing
df_fake_manual_testing = df_fake.tail(10)
for i in range(1800,1800,-1):
    df_fake.drop([i], axis = 0, inplace = True)
    
    
df_true_manual_testing = df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i], axis = 0, inplace = True)
    
df_fake.shape, df_true.shape


df_fake_manual_testing["class"] = 0
df_true_manual_testing["class"] = 1

df_fake_manual_testing.head(10)
df_true_manual_testing.head(10)


df_merge = pd.concat([df_fake, df_true], axis =0 )
df_merge.head(10)

df_merge.columns

# Delete useless columns
df = df_merge.drop(["category", "rating"], axis = 1)


df.isnull().sum()

df = df.sample(frac = 1)

df.head()

df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)

df.columns

df.head()

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


df["text_"] = df["text_"].apply(wordopt)


x = df["text_"]
y = df["class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


LR = LogisticRegression()
LR.fit(xv_train,y_train)

pred_lr=LR.predict(xv_test)
LR.score(xv_test, y_test)

print(classification_report(y_test, pred_lr))


DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

pred_dt = DT.predict(xv_test)
DT.score(xv_test, y_test)

print(classification_report(y_test, pred_dt))


GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train, y_train)

pred_gbc = GBC.predict(xv_test)
GBC.score(xv_test, y_test)

print(classification_report(y_test, pred_gbc))


RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)

pred_rfc = RFC.predict(xv_test)
RFC.score(xv_test, y_test)

print(classification_report(y_test, pred_rfc))

# Save models
joblib.dump(LR, 'models/lr_model.pkl')
joblib.dump(DT, 'models/dt_model.pkl')
joblib.dump(GBC, 'models/gbc_model.pkl')
joblib.dump(RFC, 'models/rfc_model.pkl')
joblib.dump(vectorization, 'models/vectorizer.pkl')
