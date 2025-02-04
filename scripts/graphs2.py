import pandas as pd
import matplotlib.pyplot as plt

file_path = "./data/fake_reviews_2.xlsx"
df = pd.read_excel(file_path)

count_before = df['label'].value_counts()
print("Avant équilibrage :\n", count_before)

df_fake = df[df['label'] == 'fake']
df_genuine = df[df['label'] == 'genuine']

df_genuine_sample = df_genuine.sample(n=len(df_fake), random_state=42)
df_balanced = pd.concat([df_fake, df_genuine_sample])

count_after = df_balanced['label'].value_counts()
print("\nAprès équilibrage :\n", count_after)

# Save updated file
df_balanced.to_excel("./data/fake_reviews_2_cleaned.xlsx", index=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].bar(count_before.index, count_before.values, color=['blue', 'red'])
axes[0].set_title("Avant équilibrage")
axes[0].set_ylabel("Nombre d'échantillons")

axes[1].bar(count_after.index, count_after.values, color=['blue', 'red'])
axes[1].set_title("Après équilibrage")

plt.tight_layout()
plt.show()

