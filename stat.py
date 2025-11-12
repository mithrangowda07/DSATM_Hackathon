import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("data_set.csv")

# Encode categorical columns
label_encoders = {}
for column in df.columns[1:]:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

# Define features (X) and target (y)
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Get feature importances
importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

# Display top features
print("🏆 Top Feature Importances:")
print(importances)

# Plot bar chart
plt.figure(figsize=(10,6))
importances.plot(kind='bar', color='skyblue')
plt.title("Feature Importance Ranking")
plt.ylabel("Importance Score")
plt.show()

from sklearn.feature_selection import mutual_info_classif

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Encode categorical features again
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Calculate mutual information
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

print("🔍 Mutual Information Scores:")
print(mi_series)

# Plot
plt.figure(figsize=(10,6))
mi_series.plot(kind='bar', color='orange')
plt.title("Feature Importance using Mutual Information")
plt.ylabel("MI Score")
plt.show()
