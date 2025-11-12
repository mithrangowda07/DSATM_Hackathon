import pandas as pd

df = pd.read_csv("data_set.csv")
df.drop(columns=['MultipleLines', 'PhoneService', 'gender', 'SeniorCitizen', 
 'Partner', 'Dependents', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
, inplace=True, errors='ignore')
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df.to_csv("train-final-3.csv", index=False, encoding="utf-8")
