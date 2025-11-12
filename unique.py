import pandas as pd

# ---------------------------------
# Step 1: Load your Excel file
# ---------------------------------
file_path = "test_data.xlsx"   # <-- Change to your file name
output_file = "unique_column_values.txt"

df = pd.read_excel(file_path)

print("✅ Excel file loaded successfully!")
print(f"Total Columns: {len(df.columns)}")

# ---------------------------------
# Step 2: Open text file for writing
# ---------------------------------
with open(output_file, "w", encoding="utf-8") as f:
    f.write("🔹 Unique Values in Each Column\n")
    f.write("========================================\n\n")

    # ---------------------------------
    # Step 3: Loop through each column
    # ---------------------------------
    for col in df.columns:
        f.write(f"🟩 Column: {col}\n")
        f.write("----------------------------------------\n")

        # Get unique values (including NaN)
        unique_vals = df[col].unique()

        # Write each unique value to file
        for val in unique_vals:
            f.write(f"{val}\n")

        f.write("\n\n")

print(f"✅ Unique values saved successfully to: {output_file}")
