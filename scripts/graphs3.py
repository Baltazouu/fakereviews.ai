import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le dataset final
df_final = pd.read_csv("./data/fake_reviews_final.csv")

# 🕵️ 2. Longueur moyenne des avis par catégorie (CG vs OR)
df_final['text_length'] = df_final['text_'].astype(str).apply(len)  # Calcule la longueur des avis

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_final, x='label', y='text_length', palette={'CG': 'red', 'OR': 'blue'})
plt.title("Longueur des avis en fonction du label \nCG: Fake OR: Real")
plt.xlabel("Label")
plt.ylabel("Nombre de caractères")
plt.show()

# 📈 3. Histogramme des longueurs des avis
plt.figure(figsize=(8, 5))
sns.histplot(df_final, x='text_length', hue='label', bins=30, kde=True, palette={'CG': 'red', 'OR': 'blue'})
plt.title("Distribution de la longueur des avis \nCG: Fake OR: Real")
plt.xlabel("Nombre de caractères")
plt.ylabel("Nombre d'avis")
plt.legend(title="Label", labels=['CG (Fake)', 'OR (Genuine)'])
plt.show()
