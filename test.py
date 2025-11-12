import pandas as pd

# ---------------------------------
# Step 1: Load your dataset
# ---------------------------------
file_path = "train_data.xlsx"  # <-- change this to your actual file name
df = pd.read_excel(file_path)

# ---------------------------------
# Step 2: Check if the 'Churn' column exists
# ---------------------------------
if "Churn" not in df.columns:
    raise ValueError("❌ 'Churn' column not found in the dataset!")

# ---------------------------------
# Step 3: Count churn and non-churn customers
# ---------------------------------
churn_counts = df["Churn"].value_counts(dropna=False)

print("📊 Churn Count:")
print(churn_counts)

# ---------------------------------
# Step 4: Calculate churn percentages
# ---------------------------------
churn_percent = df["Churn"].value_counts(normalize=True) * 100

print("\n📈 Churn Percentage:")
print(churn_percent)

# ---------------------------------
# Step 5: Optional - Combine both in one DataFrame for clear view
# ---------------------------------
summary = pd.DataFrame({
    'Count': churn_counts,
    'Percentage (%)': churn_percent.round(2)
})
print("\n✅ Churn Summary:")
print(summary)
