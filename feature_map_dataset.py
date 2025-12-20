import pandas as pd

df = pd.read_csv('Datasets/Lung_Cancer_4.csv')

# Feature engineering
df_transformed = pd.DataFrame({
    "gender": df["GENDER"],
    "age": df["AGE"].round().astype(int),
    "smoking": df["SMOKING"],
    "yellow_fingers": df["FINGER_DISCOLORATION"],
    "anxiety": df["MENTAL_STRESS"],
    "peer_pressure": (
        (df["SMOKING_FAMILY_HISTORY"] == 1) & (df["MENTAL_STRESS"] == 1)
    ).astype(int),
    "chronic_disease": df["LONG_TERM_ILLNESS"],
    "fatigue": (df["ENERGY_LEVEL"] < 55).astype(int),
    "allergy": df["IMMUNE_WEAKNESS"],
    "wheezing": df["BREATHING_ISSUE"],
    "alcohol": df["ALCOHOL_CONSUMPTION"],
    "coughing": df["THROAT_DISCOMFORT"],
    "shortness_of_breath": ((df["OXYGEN_SATURATION"] < 94) & (df["BREATHING_ISSUE"] == 1)).astype(int),
    "swallowing_difficulty": (
        (df["THROAT_DISCOMFORT"] == 1) & (df["BREATHING_ISSUE"] == 1)
    ).astype(int),
    "chest_pain": df["CHEST_TIGHTNESS"],
    "lung_cancer": df["PULMONARY_DISEASE"].map({"YES": 1, "NO": 0})
})

# 🔹 REMOVE NULL VALUES
df = df_transformed.dropna()

# 🔹 REMOVE DUPLICATE ROWS
df = df.drop_duplicates()

print(df.head())

df.to_csv('Datasets/Lung_Cancer_3.csv', index=False)

print((df['lung_cancer'] == 1).sum())
print((df['lung_cancer'] == 0).sum())