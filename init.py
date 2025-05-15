# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the dataset
try:
    iris = load_iris(as_frame=True)
    df = iris.frame  # Load as a pandas DataFrame
    print("Dataset loaded successfully.")
except Exception as e:
    print("Error loading the dataset:", e)

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Explore the structure of the dataset
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Clean the dataset (Iris dataset has no missing values, but let's include the step)
df.dropna(inplace=True)
print("\nAfter cleaning (if any missing values):")
print(df.info())
# Visualize the dataset
plt.figure(figsize=(10, 6))