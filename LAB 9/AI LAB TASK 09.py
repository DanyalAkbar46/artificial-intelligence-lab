

import pandas as pd
import numpy as np


# STEP 1: DATA LOADING / READING


# Load the CSV dataset
df = pd.read_csv("data/raw/Conversation.csv")

# Display first 5 rows
print("=" * 60)
print("FIRST 5 ROWS OF DATASET")
print("=" * 60)
print(df.head())

# Display last 5 rows
print("\nLAST 5 ROWS OF DATASET")
print("=" * 60)
print(df.tail())

# STEP 2: DATA EXPLORATION


# Shape of dataset (rows, columns)
print("\n" + "=" * 60)
print("DATASET SHAPE (Rows, Columns):")
print("=" * 60)
print(df.shape)

# Column names
print("\nCOLUMN NAMES:")
print("=" * 60)
print(df.columns.tolist())

# Data types of each column
print("\nDATA TYPES:")
print("=" * 60)
print(df.dtypes)

# Basic statistical summary
print("\nSTATISTICAL SUMMARY:")
print("=" * 60)
print(df.describe(include='all'))

# Count of null/missing values
print("\nNULL VALUES COUNT PER COLUMN:")
print("=" * 60)
print(df.isnull().sum())

# Total number of missing values
print("\nTOTAL MISSING VALUES:", df.isnull().sum().sum())

# Unique values in each column
print("\nUNIQUE VALUES PER COLUMN:")
print("=" * 60)
for col in df.columns:
    print(f"  {col}: {df[col].nunique()} unique values")

# Sample random rows
print("\nRANDOM SAMPLE (5 rows):")
print("=" * 60)
print(df.sample(5))

# Value counts for first column
print("\nVALUE COUNTS (first column):")
print("=" * 60)
print(df.iloc[:, 0].value_counts().head(10))

print("\n Lab 9 Complete: Data Loading and Exploration Done!")
