# iris_data_analysis.py

# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


try:
    iris = load_iris(as_frame=True)
    df = iris.frame  # Load as a pandas DataFrame
    print("✅ Dataset loaded successfully.")
except Exception as e:
    print("❌ Error loading the dataset:", e)

# Display the first few rows
print("\n🔹 First 5 rows of the dataset:")
print(df.head())

# Explore the structure of the dataset
print("\n🔹 Dataset Info:")
print(df.info())

# Check for missing values
print("\n🔹 Missing values in each column:")
print(df.isnull().sum())

# Clean the dataset (drop rows with missing values if any)
df.dropna(inplace=True)
print("\n✅ Dataset cleaned (if needed):")
print(df.info())


# Compute basic statistics
print("\n🔸 Descriptive statistics:")
print(df.describe())

# Group by target (numeric class) and compute mean
grouped = df.groupby('target').mean()
print("\n🔸 Mean values by target (species index):")
print(grouped)

# Map numeric target to species name
df['species'] = df['target'].map(dict(enumerate(iris.target_names)))

# Group by species and compute mean
print("\n🔸 Mean values grouped by species name:")
print(df.groupby('species').mean())

# Display a finding
print("\n🔍 Insight:")
print("Iris-virginica has the highest average petal length and width, while Iris-setosa has the smallest.")


# Set style
sns.set(style="whitegrid")

#  Line Chart - Simulating time trend by sample index
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
plt.title('Sepal Length over Sample Index')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.tight_layout()
plt.show()

# Bar Chart - Average petal length per species
plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='species', y='petal length (cm)', palette='muted')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.show()

#  Histogram - Distribution of sepal width
plt.figure(figsize=(8, 6))
sns.histplot(df['sepal width (cm)'], bins=15, kde=True, color='skyblue')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

#  Scatter Plot - Sepal Length vs. Petal Length
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='deep')
plt.title('Sepal Length vs. Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.tight_layout()
plt.show()
