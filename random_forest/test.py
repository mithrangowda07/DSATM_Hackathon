import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import joblib

# -------------------------------
# Step 1: Load Trained Model
# -------------------------------
model_path = r"D:\Projects\DSATM_Hackathon\random_forest\random_forest_model.pkl"
model = joblib.load(model_path)
print("✅ Random Forest model loaded successfully!")

# -------------------------------
# Step 2: Load Test Dataset
# -------------------------------
test_file = r"D:\Projects\DSATM_Hackathon\test1_data.xlsx"  # <-- your test Excel file path
df = pd.read_excel(test_file)
print(f"✅ Test dataset loaded! Shape: {df.shape}")

# -------------------------------
# Step 3: Drop ID column if present
# -------------------------------
# df = df.iloc[:, 1:]  # assumes first column is ID

# -------------------------------
# Step 4: Encode Categorical Columns
# -------------------------------
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

print("🔠 All categorical columns encoded successfully!")

# -------------------------------
# Step 5: Split features & target
# -------------------------------
X_test = df.drop("Churn", axis=1)
y_test = df["Churn"]

# -------------------------------
# Step 6: Make Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Step 7: Evaluate Performance
# -------------------------------
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
report = classification_report(y_test, y_pred, target_names=["No", "Yes"])

print("\n📊 Confusion Matrix:")
print(cm)
print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")
print(f"❌ Error Rate: {error_rate * 100:.2f}%")

print("\n📈 Classification Report:")
print(report)

# -------------------------------
# Step 8: Save All Predictions + Evaluation
# -------------------------------
output_predictions_file = r"D:\Projects\DSATM_Hackathon\rf_predictions_output.xlsx"
output_report_file = r"D:\Projects\DSATM_Hackathon\rf_test_report.txt"

# Save predictions alongside test data
df["Predicted_Churn"] = np.where(y_pred == 1, "Yes", "No")
df.to_excel(output_predictions_file, index=False)
print(f"\n💾 Predictions saved to Excel: {output_predictions_file}")

# Save evaluation metrics in a text report
with open(output_report_file, "w", encoding="utf-8") as f:
    f.write("📊 Random Forest Model Evaluation Report\n")
    f.write("=========================================\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Error Rate: {error_rate * 100:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print(f"💾 Evaluation metrics saved to: {output_report_file}")
