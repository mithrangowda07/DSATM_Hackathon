import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

# -------------------------------
# Step 1: Load the dataset
# -------------------------------
file_path = "train_data.xlsx"  # <-- Change to your actual file
df = pd.read_excel(file_path)

print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# -------------------------------
# Step 2: Drop ID column if present
# -------------------------------
df = df.iloc[:, 1:]  # Removes first column (Customer ID)

# -------------------------------
# Step 3: Encode categorical (non-numeric) columns
# -------------------------------
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

print("🔠 All categorical columns encoded successfully.")

# -------------------------------
# Step 4: Define features (X) and target (y)
# -------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Ensure no NaN in numeric columns
X = X.fillna(X.median())

# Split data to avoid overfitting when evaluating feature importance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Step 5: Random Forest Feature Importance
# -------------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)

rf_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
rf_importance = rf_importance.sort_values(ascending=False)

# -------------------------------
# Step 6: Mutual Information (Statistical Measure)
# -------------------------------
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_importance = pd.Series(mi_scores, index=X.columns)
mi_importance = mi_importance.sort_values(ascending=False)

# -------------------------------
# Step 7: Combine & Compare Both Methods
# -------------------------------
comparison = pd.DataFrame({
    'RandomForest Importance': rf_importance,
    'Mutual Information Score': mi_importance
}).sort_values(by='RandomForest Importance', ascending=False)

comparison['Average Rank'] = (
    comparison['RandomForest Importance'].rank(ascending=False) +
    comparison['Mutual Information Score'].rank(ascending=False)
) / 2

comparison = comparison.sort_values(by='Average Rank')

# -------------------------------
# Step 8: Display Results
# -------------------------------
print("\n🏆 Top 10 Promising Attributes (Combined Importance):")
print(comparison.head(10).round(4))

# -------------------------------
# Step 9: Visualization
# -------------------------------
plt.figure(figsize=(10,6))
comparison.head(10)[['RandomForest Importance', 'Mutual Information Score']].plot(
    kind='bar',
    figsize=(10,6),
    color=['teal', 'orange']
)
plt.title("Top 10 Promising Features (Random Forest vs Mutual Information)")
plt.xlabel("Attributes")
plt.ylabel("Importance Score")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
