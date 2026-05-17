import pandas as pd

# Lire dataset
df = pd.read_csv("hmnist_28_28_RGB.csv")

# Chouf nombre dyal samples f kol class
print(df["label"].value_counts())

# Threshold

min_samples = 600

# Classes li عندهم >= 50
valid_classes = df["label"].value_counts()
valid_classes = valid_classes[valid_classes >= min_samples].index

# Garder ghir had classes
df_filtered = df[df["label"].isin(valid_classes)]

# Vérification
print(df_filtered["label"].value_counts())

# Save nouveau dataset
df_filtered.to_csv("hmnist_filtered.csv", index=False)

print("Dataset filtered sauvegardé.")
