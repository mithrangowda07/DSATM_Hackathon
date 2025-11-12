import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data_set.csv")  # 🔁 Replace with your actual file name or path

# Optional: Clean TotalCharges if stored as string
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# Set style
sns.set(style="whitegrid")

# 1️⃣ Histogram for MonthlyCharges
plt.figure(figsize=(8,5))
sns.histplot(df["MonthlyCharges"], bins=30, kde=True, color='steelblue')
plt.title("Distribution of Monthly Charges")
plt.xlabel("Monthly Charges")
plt.ylabel("Number of Customers")
plt.show()

# 2️⃣ Histogram for TotalCharges
plt.figure(figsize=(8,5))
sns.histplot(df["TotalCharges"], bins=30, kde=True, color='green')
plt.title("Distribution of Total Charges")
plt.xlabel("Total Charges")
plt.ylabel("Number of Customers")
plt.show()

# 3️⃣ Box Plot for MonthlyCharges and TotalCharges
plt.figure(figsize=(8,5))
sns.boxplot(data=df[["MonthlyCharges", "TotalCharges"]], palette="pastel")
plt.title("Boxplot of Monthly and Total Charges")
plt.ylabel("Charge Amount")
plt.show()

# 4️⃣ Scatter Plot: MonthlyCharges vs TotalCharges
plt.figure(figsize=(8,6))
sns.scatterplot(x="MonthlyCharges", y="TotalCharges", data=df, alpha=0.6)
plt.title("Monthly Charges vs Total Charges")
plt.xlabel("Monthly Charges")
plt.ylabel("Total Charges")
plt.show()

# 5️⃣ (Optional) Compare by Churn Status
if "Churn" in df.columns:
    plt.figure(figsize=(8,5))
    sns.boxplot(x="Churn", y="MonthlyCharges", data=df, palette="coolwarm")
    plt.title("Monthly Charges vs Churn Status")
    plt.show()

    plt.figure(figsize=(8,5))
    sns.boxplot(x="Churn", y="TotalCharges", data=df, palette="coolwarm")
    plt.title("Total Charges vs Churn Status")
    plt.show()
