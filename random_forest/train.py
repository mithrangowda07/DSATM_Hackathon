import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib

# -------------------------------
# Step 1: Load dataset
# -------------------------------
file_path = r"D:\Projects\DSATM_Hackathon\train_data.xlsx"
df = pd.read_excel(file_path)
# df = df.iloc[:, 1:]  # drop CustomerID

# Encode categorical columns
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

X = df.drop("Churn", axis=1)
y = df["Churn"]

# -------------------------------
# Step 2: Handle imbalance using SMOTE
# -------------------------------
print("⚖️ Applying SMOTE to balance classes...")
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

print(f"✅ Before SMOTE: {np.bincount(y)}")
print(f"✅ After SMOTE:  {np.bincount(y_resampled)}")

# -------------------------------
# Step 3: Split train and test data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# -------------------------------
# Step 4: Train Random Forest
# -------------------------------
model = RandomForestClassifier(
    n_estimators=300,          # more trees → more stable
    max_depth=None,           # let trees expand fully
    min_samples_split=5,      # control overfitting
    min_samples_leaf=3,       # avoid tiny leaves
    class_weight='balanced',  # handles remaining imbalance
    random_state=42,
    n_jobs=-1                 # use all CPU cores
)

model.fit(X_train, y_train)
print("\n🌳 Random Forest model trained successfully!")

# -------------------------------
# Step 5: Evaluate model
# -------------------------------
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# --- Training performance ---
train_acc = accuracy_score(y_train, y_pred_train)
print(f"\n📈 Training Accuracy: {train_acc * 100:.2f}%")

# --- Testing performance ---
cm = confusion_matrix(y_test, y_pred_test)
test_acc = accuracy_score(y_test, y_pred_test)
error_rate = 1 - test_acc

print("\n📊 Confusion Matrix (Test Data):")
print(cm)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")
print(f"❌ Error Rate: {error_rate * 100:.2f}%")

print("\n📄 Classification Report (Test Data):")
print(classification_report(y_test, y_pred_test))

# -------------------------------
# Step 6: Save trained model
# -------------------------------
model_filename = r"D:\Projects\DSATM_Hackathon\random_forest_model.pkl"
joblib.dump(model, model_filename)
print(f"\n💾 Model saved as: {model_filename}")
