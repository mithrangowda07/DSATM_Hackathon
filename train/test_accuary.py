import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# ---------------------------------
# Step 1: Load actual churn values from Excel
# ---------------------------------
excel_file = r"D:\Projects\DSATM_Hackathon\test1_data.xlsx"   # Excel file with real churn column
df_actual = pd.read_excel(excel_file)

# Assuming the actual churn column is named "Churn"
actual = df_actual["Churn"].astype(str).str.strip().str.title()  # normalize case (Yes/No)

print(f"✅ Loaded actual churn values from Excel: {len(actual)} rows")

# ---------------------------------
# Step 2: Load model predictions from text file
# ---------------------------------
pred_file = r"D:\Projects\DSATM_Hackathon\train\churn_predictions.txt"
with open(pred_file, "r") as f:
    predictions = [line.strip().title() for line in f.readlines()]

print(f"✅ Loaded model predictions from TXT: {len(predictions)} rows")

# ---------------------------------
# Step 3: Validate that lengths match
# ---------------------------------
if len(actual) != len(predictions):
    raise ValueError(f"❌ Mismatch: Excel has {len(actual)} rows, but TXT has {len(predictions)} predictions")

# ---------------------------------
# Step 4: Encode Yes/No to 1/0
# ---------------------------------
mapping = {"Yes": 1, "No": 0}
actual_encoded = actual.map(mapping)
pred_encoded = pd.Series(predictions).map(mapping)

# ---------------------------------
# Step 5: Compute metrics
# ---------------------------------
cm = confusion_matrix(actual_encoded, pred_encoded)
accuracy = accuracy_score(actual_encoded, pred_encoded)
error_rate = 1 - accuracy
report = classification_report(actual_encoded, pred_encoded, target_names=["No", "Yes"])

# ---------------------------------
# Step 6: Display results
# ---------------------------------
print("\n📊 Confusion Matrix:")
print(cm)

print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")
print(f"❌ Error Rate: {error_rate * 100:.2f}%")

print("\n📈 Classification Report:")
print(report)

# ---------------------------------
# Step 7: Save results to text file
# ---------------------------------
output_file = r"D:\Projects\DSATM_Hackathon\train\model_evaluation_report.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("📊 Model Evaluation Report\n")
    f.write("====================================\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Error Rate: {error_rate * 100:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")

print(f"\n💾 Evaluation results saved to: {output_file}")

