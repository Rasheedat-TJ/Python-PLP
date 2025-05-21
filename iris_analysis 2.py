
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Dataset structure
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Basic statistics
print("\nStatistical Summary:")
print(df.describe())

# Group by species and compute mean
print("\nMean of features grouped by species:")
print(df.groupby('species').mean())

# Visualization: Pair plot
sns.pairplot(df, hue='species')
plt.suptitle("Iris Dataset Pairplot", y=1.02)
plt.savefig("pairplot.png")
plt.close()

# Box plot of petal length by species
plt.figure(figsize=(8, 5))
sns.boxplot(x='species', y='petal_length', data=df)
plt.title("Petal Length by Species")
plt.savefig("boxplot_petal_length.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

print("\nVisualizations saved as 'pairplot.png', 'boxplot_petal_length.png', and 'correlation_heatmap.png'.")
