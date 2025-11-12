import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

# ---------------------------------
# Step 1: Load the trained model
# ---------------------------------
model_path = r"D:\Projects\DSATM_Hackathon\train\decision_tree_model.pkl"
model = joblib.load(model_path)

print("✅ Model loaded successfully!")

# ---------------------------------
# Step 2: Load the new test data
# ---------------------------------
test_file = r"D:\Projects\DSATM_Hackathon\test_data.xlsx"   # <-- change to your test Excel file
df_test = pd.read_excel(test_file)

print(f"✅ Test data loaded successfully! Shape: {df_test.shape}")

# ---------------------------------
# Step 3: Drop Customer ID column if present
# ---------------------------------
df_test = df_test.iloc[:, 1:]  # assuming first column is Customer ID

# ---------------------------------
# Step 4: Encode categorical columns (same as training)
# ---------------------------------
for col in df_test.columns:
    if df_test[col].dtype == 'object':
        le = LabelEncoder()
        df_test[col] = le.fit_transform(df_test[col].astype(str))

print("🔠 Test data encoded successfully!")

# ---------------------------------
# Step 5: Make predictions using the loaded model   
# ---------------------------------
predictions = model.predict(df_test)

# Map numeric output to 'Yes'/'No'
pred_labels = np.where(predictions == 1, "Yes", "No")

# ---------------------------------
# Step 6: Save predictions to a text file
# ---------------------------------
output_file = r"D:\Projects\DSATM_Hackathon\churn_predictions.txt"
with open(output_file, "w") as f:
    for label in pred_labels:
        f.write(label + "\n")

print(f"💾 Predictions saved to: {output_file}")
print("\nSample Predictions:")
print(pred_labels[:10])
