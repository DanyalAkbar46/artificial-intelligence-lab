=

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


import pandas as pd
import numpy as np
import re

# Load dataset (continuing from Lab 9)
df = pd.read_csv("data/raw/Conversation.csv")

print("=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)
print(df.head())
print(f"\nShape: {df.shape}")
print(f"Data Types:\n{df.dtypes}")

# STEP 1: DATA PRE-PROCESSING


print("\n" + "=" * 60)
print("STEP 1: DATA PRE-PROCESSING")
print("=" * 60)

# Remove duplicate rows
before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"Duplicates removed: {before - after} rows")

# Reset index after dropping duplicates
df = df.reset_index(drop=True)

# Clean text columns — lowercase, remove special characters
def clean_text(text):
    if pd.isnull(text):
        return text
    text = str(text).lower()                          # lowercase
    text = re.sub(r'http\S+|www\S+', '', text)        # remove URLs
    text = re.sub(r'\S+@\S+', '', text)               # remove emails
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)        # remove special chars
    text = re.sub(r'\s+', ' ', text).strip()          # remove extra spaces
    return text

# Apply cleaning to text columns
text_columns = df.select_dtypes(include='object').columns
for col in text_columns:
    df[col] = df[col].apply(clean_text)
    print(f"  ✓ Cleaned column: '{col}'")

print("\nAfter Preprocessing:")
print(df.head())

# STEP 2: DEALING WITH NULL VALUES


print("\n" + "=" * 60)
print("STEP 2: DEALING WITH NULL VALUES")
print("=" * 60)

# Check null values before handling
print("Null values BEFORE handling:")
print(df.isnull().sum())

# Strategy: Fill null values in text columns with empty string
for col in df.select_dtypes(include='object').columns:
    null_count = df[col].isnull().sum()
    if null_count > 0:
        df[col].fillna("unknown", inplace=True)
        print(f"   Filled {null_count} nulls in '{col}' with 'unknown'")

# Strategy: Fill null values in numeric columns with 0
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    null_count = df[col].isnull().sum()
    if null_count > 0:
        df[col].fillna(0, inplace=True)
        print(f"   Filled {null_count} nulls in '{col}' with 0")

# Drop any remaining rows that still have nulls (if any)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print("\nNull values AFTER handling:")
print(df.isnull().sum())
print(f"\nDataset shape after null handling: {df.shape}")

# STEP 3: CHANGING DATATYPES TO INT


print("\n" + "=" * 60)
print("STEP 3: CHANGING DATATYPES TO INT")
print("=" * 60)

print("Data types BEFORE conversion:")
print(df.dtypes)

# Add a length column (numeric) for demonstration
df['question_length'] = df['question'].apply(lambda x: len(str(x)))
df['answer_length']   = df['answer'].apply(lambda x: len(str(x)))

# The unnamed index column (if present) convert to int
if 'Unnamed: 0' in df.columns:
    df['Unnamed: 0'] = df['Unnamed: 0'].astype(int)
    print("  ✓ Converted 'Unnamed: 0' to Int")

# Convert newly created length columns to int
df['question_length'] = df['question_length'].astype(int)
df['answer_length']   = df['answer_length'].astype(int)
print("  ✓ Converted 'question_length' to Int")
print("  ✓ Converted 'answer_length'   to Int")

print("\nData types AFTER conversion:")
print(df.dtypes)

print("\nFinal Dataset Preview:")
print(df.head())
print(f"\nFinal Shape: {df.shape}")

# Save preprocessed data
df.to_csv("data/processed/Conversation_Preprocessed.csv", index=False)
print("\n Preprocessed data saved to: data/processed/Conversation_Preprocessed.csv")


