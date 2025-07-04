import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load Titanic dataset
df = sns.load_dataset("titanic")
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Drop rows with nulls
numeric_df = numeric_df.dropna()

# Compute correlation matrix
correlation_matrix = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix Heatmap - Titanic Dataset")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()
