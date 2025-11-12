import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------------
# Step 1: Load the dataset
# -------------------------------
file_path = "data_set.csv"   # <-- change this to your actual file name
df = pd.read_csv(file_path)

print("✅ Dataset loaded successfully!")
print("Total rows:", len(df))
print("Columns:", df.columns.tolist())

# -------------------------------
# Step 2: Drop the Customer ID column (if present)
# -------------------------------
df = df.iloc[:, 1:]   # removes the first column (ID)

# -------------------------------
# Step 3: Define features (X) and target (y)
# -------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

# -------------------------------
# Step 4: Split the dataset (80% training, 20% testing)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% data for testing
    random_state=42,     # ensures same split every time
    stratify=y           # keeps same churn ratio in both sets
)

# -------------------------------
# Step 5: Combine X and y back for saving
# -------------------------------
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# -------------------------------
# Step 6: Save to Excel files
# -------------------------------
train_data.to_excel("train_data.xlsx", index=False)
test_data.to_excel("test_data.xlsx", index=False)

print("\n✅ Data successfully split and saved!")
print("Training set size:", train_data.shape)
print("Testing set size:", test_data.shape)
print("\nFiles created: 'train_data.xlsx' and 'test_data.xlsx'")
