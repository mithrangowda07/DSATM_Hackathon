import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import joblib

# -------------------------------
# Step 1: Load dataset
# -------------------------------
file_path = r"D:\Projects\DSATM_Hackathon\train_data.xlsx"
df = pd.read_excel(file_path)
# df = df.iloc[:, 1:]  # drop ID

# Encode categorical columns
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------
# Step 2: Train the Decision Tree
# ---------------------------------
model = DecisionTreeClassifier(
    criterion='gini',         # 'entropy' if you want info gain
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

print("\n🌳 Decision Tree trained successfully!")
print(f"Impurity Measure Used: {model.criterion.upper()}")

# ---------------------------------
# Step 3: Evaluate the Model
# ---------------------------------
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n📊 Confusion Matrix:")
print(cm)

# Accuracy and Error Rate
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")
print(f"❌ Error Rate: {error_rate * 100:.2f}%")

# Classification Report
print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred))

# ---------------------------------
# Step 4: Root Node Details
# ---------------------------------
tree = model.tree_
root_index = 0
root_feature_index = tree.feature[root_index]
root_impurity = tree.impurity[root_index]
root_samples = tree.n_node_samples[root_index]
root_value = tree.value[root_index]

if root_feature_index == -2:
    root_feature = "Leaf Node (no split)"
else:
    root_feature = X.columns[root_feature_index]

print("\n🌲 Root Node (First Split) Details:")
print(f"Feature used for first split: {root_feature}")
print(f"Impurity at root node: {root_impurity:.4f}")
print(f"Samples at root node: {root_samples}")
print(f"Class distribution at root: {root_value}")

# ---------------------------------
# Step 5: Impurity & Information Gain for Each Feature
# ---------------------------------
def impurity_measure(y_values, criterion='entropy'):
    classes, counts = np.unique(y_values, return_counts=True)
    probs = counts / counts.sum()
    if criterion == 'gini':
        return 1 - np.sum(probs ** 2)
    elif criterion == 'entropy':
        return -np.sum(probs * np.log2(probs + 1e-9))
    elif criterion == 'log_loss':
        return -np.sum(probs * np.log(probs + 1e-9))
    else:
        raise ValueError("Invalid criterion")

root_impurity_custom = impurity_measure(y_train, criterion=model.criterion)
print(f"\n🔹 Overall impurity at root (before any split): {root_impurity_custom:.4f}\n")

impurity_after_split = {}
for feature in X_train.columns:
    feature_values = X_train[feature].unique()
    weighted_impurity = 0
    for val in feature_values:
        subset = y_train[X_train[feature] == val]
        if len(subset) > 0:
            impurity_val = impurity_measure(subset, criterion=model.criterion)
            weighted_impurity += (len(subset) / len(y_train)) * impurity_val
    impurity_after_split[feature] = weighted_impurity

results = pd.DataFrame({
    'Feature': impurity_after_split.keys(),
    'Weighted_Impurity': impurity_after_split.values()
})
results['Information_Gain'] = root_impurity_custom - results['Weighted_Impurity']
results = results.sort_values(by='Information_Gain', ascending=False).reset_index(drop=True)

print("🏆 Impurity and Information Gain for Each Attribute at Root Node:")
print(results.round(4))

# ---------------------------------
# Step 6: Save Model
# ---------------------------------
model_filename = r"D:\Projects\DSATM_Hackathon\decision_tree_model.pkl"
joblib.dump(model, model_filename)
print(f"\n💾 Model saved successfully as: {model_filename}")
