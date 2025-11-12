import pandas as pd
from sklearn.model_selection import train_test_split

# --- Step 1: Load CSV (handles encoding issues) ---
try:
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv", encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv", encoding="latin1")

print("✅ Data loaded successfully! Rows:", len(df))

# --- Step 2: Clean & Prepare ---
if "customerID" in df.columns:
    df.rename(columns={"customerID": "id"}, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(subset=["TotalCharges"], inplace=True)

# --- Step 3: Split into train/test ---
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Churn"])

train_df.to_csv("train_ready.csv", index=False, encoding="utf-8")
test_df.to_csv("test_ready.csv", index=False, encoding="utf-8")

print("\n✅ Split complete!")
print("train_ready.csv →", len(train_df), "rows")
print("test_ready.csv  →", len(test_df), "rows")
