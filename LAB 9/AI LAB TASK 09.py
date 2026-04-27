import pandas as pd

# Load Titanic dataset
df = pd.read_csv("titanic.csv")

# Display first 5 rows
print(df.head())
# Dataset shape (rows, columns)
print("Shape:", df.shape)

# Column names
print("Columns:", df.columns)

# Info about data types and missing values
print(df.info())

# Summary statistics for numerical columns
print(df.describe())

# Missing values count
print(df.isnull().sum())

# Explore categorical values
print(df['Sex'].value_counts())
print(df['Survived'].value_counts())
