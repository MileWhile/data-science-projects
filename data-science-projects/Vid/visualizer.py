import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load the Titanic dataset
df = sns.load_dataset("titanic")

# Drop rows with null values for clean visuals
df = df.dropna(subset=["age", "fare", "sex", "class", "survived"])

# -------- Bar Chart: Survival Count --------
plt.figure(figsize=(6, 4))
sns.countplot(x='survived', data=df)
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("bar_chart_survival.png")  # Save image
plt.show()

# -------- Scatter Plot: Age vs Fare --------
plt.figure(figsize=(6, 4))
sns.scatterplot(x='age', y='fare', hue='sex', data=df)
plt.title("Age vs Fare Paid")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.legend(title='Sex')
plt.tight_layout()
plt.savefig("scatter_plot_age_fare.png")
plt.show()

# -------- Pie Chart: Class Distribution --------
class_counts = df['class'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Passenger Class Distribution")
plt.tight_layout()
plt.savefig("pie_chart_class_distribution.png")
plt.show()
