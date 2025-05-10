import pandas as pd

file = pd.read_csv("../data/kaggle_unclean_dataset.csv")

# Map symptoms to binary
binary_map = {"Yes": 1, "No": 0}
for col in ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]:
    file[col] = file[col].map(binary_map)

# Change unique disease to ints
disease_labels = {disease: idx for idx, disease in enumerate(file["Disease"].unique())}
file["Condition"] = file["Disease"].map(disease_labels)

# Keep columns we want
file = file[["Fever", "Cough", "Fatigue", "Difficulty Breathing", "Condition"]]

# Save new cleaned file
file.to_csv("../data/kaggle_clean_dataset.csv", index=False)
disease_labels = {value: key for key, value in disease_labels.items()}
print("Disease label mapping:")
for key, value in disease_labels.items():
    print(f"{key}: '{value}',")
