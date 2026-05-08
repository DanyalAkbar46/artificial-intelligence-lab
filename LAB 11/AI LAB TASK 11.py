
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
        print(f"  ✓ Filled {null_count} nulls in '{col}' with 'unknown'")

# Strategy: Fill null values in numeric columns with 0
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    null_count = df[col].isnull().sum()
    if null_count > 0:
        df[col].fillna(0, inplace=True)
        print(f"  ✓ Filled {null_count} nulls in '{col}' with 0")

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


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

# LOAD PREPROCESSED DATA (from Lab 10)

print("=" * 60)
print("LOADING PREPROCESSED DATA")
print("=" * 60)

df = pd.read_csv("data/processed/Conversation_Preprocessed.csv")
print(df.head())
print(f"Shape: {df.shape}")

# PREPARE FEATURES AND LABELS


# This creates a binary classification task from our dataset

df['label'] = df['answer_length'].apply(lambda x: 1 if x > 50 else 0)

print("\nLabel Distribution:")
print(df['label'].value_counts())
print(f"  0 = Short answer (<=50 chars)")
print(f"  1 = Long  answer (>50 chars)")

X = df['question']   # Feature: question text
y = df['label']      # Target:  short(0) or long(1) answer

# STEP 1: TRAIN / TEST SPLITTING

print("\n" + "=" * 60)
print("STEP 1: TRAIN / TEST SPLITTING (80% Train, 20% Test)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"Total Samples  : {len(df)}")
print(f"Training Set   : {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"Testing Set    : {len(X_test)}  samples ({len(X_test)/len(df)*100:.1f}%)")

# STEP 2: CHOOSE AND APPLY THE RIGHT MODEL

print("\n" + "=" * 60)
print("STEP 2: CHOOSE AND APPLY MODEL")
print("=" * 60)

# Model chosen: Naive Bayes (MultinomialNB)


print("Model Selected: Naive Bayes (MultinomialNB)")
print("Reason: Ideal for text/NLP classification problems")

# Convert text to TF-IDF numeric features
# (Machine Learning models need numbers, not raw text)
vectorizer = TfidfVectorizer(max_features=500)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

print(f"\nTF-IDF Feature Matrix Shape (Train): {X_train_vec.shape}")
print(f"TF-IDF Feature Matrix Shape (Test) : {X_test_vec.shape}")

# Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)
print("\n✓ Model trained successfully!")

# STEP 3: TESTING / PREDICTING

print("\n" + "=" * 60)
print("STEP 3: TESTING / PREDICTING")
print("=" * 60)

y_pred = model.predict(X_test_vec)

# Show first 10 predictions vs actual
print("Sample Predictions (first 10):")
print(f"{'Question':<45} {'Actual':<10} {'Predicted':<10}")
print("-" * 65)
for i in range(min(10, len(X_test))):
    q = list(X_test)[i][:40] + "..."
    actual    = list(y_test)[i]
    predicted = y_pred[i]
    status = "✓" if actual == predicted else "✗"
    print(f"{q:<45} {actual:<10} {predicted:<10} {status}")

# STEP 4: DISPLAY ACCURACY SCORE

print("\n" + "=" * 60)
print("STEP 4: ACCURACY SCORE")
print("=" * 60)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")

print("\nDetailed Classification Report:")
print("-" * 60)
print(classification_report(y_test, y_pred,
      target_names=["Short Answer (0)", "Long Answer (1)"]))

print("Confusion Matrix:")
print("-" * 60)
cm = confusion_matrix(y_test, y_pred)
print(f"               Predicted 0    Predicted 1")
print(f"  Actual 0  :     {cm[0][0]:<10}    {cm[0][1]}")
print(f"  Actual 1  :     {cm[1][0]:<10}    {cm[1][1]}")

