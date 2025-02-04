import pandas as pd
import matplotlib.pyplot as plt

def plot_pie_chart(path, column, labels, title, ax,type):
    df = None
    if(type == "csv"): 
        df = pd.read_csv(path, encoding="latin1")
    elif(type == "excel"):
        df = pd.read_excel(path)
        
    data_counts = df[column].value_counts().reindex(labels.keys(), fill_value=0)
    ax.pie(
        data_counts, labels=labels.values(),
        autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'], startangle=140
    )
    ax.set_title(title)

# Création des sous-graphiques
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

plot_pie_chart(
    "./data/fake_reviews.csv", "label", 
    {"OR": "Original Review (OR)", "CG": "Computer Generated (CG)"},
    "Répartition des types de données (OR vs CG)\nDataset : fake_reviews.csv", axes[0],
    "csv"
)

plot_pie_chart(
    "./data/fake_reviews_2.xlsx", "label", 
    {"fake": "Computer Generated (0)", "genuine": "Original Review (1)"},
    "Répartition des types de données (0 vs 1)\nDataset : fake_reviews_2.csv", axes[1],
    "excel"
)

plt.tight_layout()
plt.show()
