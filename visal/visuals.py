# Let's generate multiple visualizations that would suit the user's churn dataset.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

# Load dataset
file_path = r"churn.csv"
df = pd.read_csv(file_path)

# Drop ID column if present
df = df.iloc[:, 1:]

# Encode categorical columns
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# Basic info
X = df.drop("Churn", axis=1)
y = df["Churn"]

# EDA Visualization 1: Churn Distribution Pie Chart
plt.figure(figsize=(6,6))
df["Churn"].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=["#66b3ff","#ff9999"], labels=["No", "Yes"])
plt.title("Churn Distribution (Yes vs No)")
plt.ylabel("")
plt.tight_layout()
plt.show()

# EDA Visualization 2: Churn by Contract Type
plt.figure(figsize=(7,5))
sns.countplot(x="Contract", hue="Churn", data=df, palette="Set2")
plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# EDA Visualization 3: Tenure Distribution by Churn
plt.figure(figsize=(7,5))
sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", palette="coolwarm", bins=30)
plt.title("Tenure Distribution by Churn")
plt.xlabel("Tenure (months)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# EDA Visualization 4: Boxplot - MonthlyCharges vs Churn
plt.figure(figsize=(7,5))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df, palette="coolwarm")
plt.title("Monthly Charges vs Churn")
plt.tight_layout()
plt.show()

# EDA Visualization 5: Correlation Heatmap
plt.figure(figsize=(10,6))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
