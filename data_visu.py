import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno

# ------------------------------
# Step 1: Load the Excel dataset
# ------------------------------
file_path = "data_set.csv"   # 🔁 Replace with your actual file name or path
df = pd.read_csv(file_path)

# Display first few rows
print("✅ Dataset loaded successfully!")
print(df.head())

# ------------------------------
# Step 2: Ignore the first column (Customer ID)
# ------------------------------
columns_to_analyze = df.columns[1:]  # skip first column

# ------------------------------
# Step 3: Show summary info
# ------------------------------
print("\n📋 Dataset Info:")
print(df.info())

# ------------------------------
# Step 4: Visualize missing values
# ------------------------------
print("\n📉 Missing Values Overview:")
msno.matrix(df)
plt.show()

# ------------------------------
# Step 5: Loop through each column (except first)
# ------------------------------
for col in columns_to_analyze:
    print(f"\n🔹 Column: {col}")
    
    # Show value counts (including NaN)
    print(df[col].value_counts(dropna=False))
    
    # Plot bar chart
    plt.figure(figsize=(8, 4))
    df[col].value_counts(dropna=False).plot(kind='bar')
    plt.title(f"Value Distribution for {col}")
    plt.xlabel("Values")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
