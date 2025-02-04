import pandas as pd

file_path_xlsx = "./data/fake_reviews_2_cleaned.xlsx"
df_xlsx = pd.read_excel(file_path_xlsx)

file_path_csv = "./data/fake_reviews.csv"
df_csv = pd.read_csv(file_path_csv)

df_xlsx = df_xlsx[['label', 'review']].rename(columns={'review': 'text_'})  
df_csv = df_csv[['label', 'text_']] 

df_xlsx['label'] = df_xlsx['label'].map({'fake': 'CG', 'genuine': 'OR'})

print("XLSX :")
print(df_xlsx['label'].value_counts())

print("\nCSV :")
print(df_csv['label'].value_counts())


# Merge datasets
df_final = pd.concat([df_xlsx, df_csv], ignore_index=True)

# save to csv
df_final.to_csv("./data/fake_reviews_final.csv", index=False)

print("Fusion et conversion terminées avec succès ! 🚀")
