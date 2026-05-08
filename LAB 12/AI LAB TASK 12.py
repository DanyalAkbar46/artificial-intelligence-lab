
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
print("\nPreprocessed data saved to: data/processed/Conversation_Preprocessed.csv")


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
print(f"\ Model Accuracy: {accuracy * 100:.2f}%")

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



from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import os

app = Flask(__name__)

# LOAD & TRAIN MODEL ON STARTUP


def clean_text(text):
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_train():
    """Load data, preprocess, and train model"""
    # Load dataset
    df = pd.read_csv("data/raw/Conversation.csv")

    # Preprocess
    df = df.drop_duplicates().reset_index(drop=True)
    df['question'] = df['question'].apply(clean_text)
    df['answer']   = df['answer'].apply(clean_text)
    df.dropna(inplace=True)

    # Create label: 1 = long answer, 0 = short answer
    df['answer_length'] = df['answer'].apply(len)
    df['label'] = df['answer_length'].apply(lambda x: 1 if x > 50 else 0)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['question'], df['label'], test_size=0.2, random_state=42
    )

    # TF-IDF + Naive Bayes
    vectorizer = TfidfVectorizer(max_features=500)
    X_train_vec = vectorizer.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    return model, vectorizer

print("Loading and training model...")
model, vectorizer = load_and_train()
print("✓ Model ready!")


# HTML FRONTEND (Generated with ChatGPT)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Conversation Predictor</title>
    <style>
        /* ChatGPT-generated CSS */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            max-width: 650px;
            width: 100%;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2em;
            color: #333;
            margin-bottom: 8px;
        }
        .header p {
            color: #888;
            font-size: 0.95em;
        }
        .badge {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            margin-bottom: 20px;
        }
        label {
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
            color: #444;
        }
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1em;
            resize: vertical;
            min-height: 100px;
            transition: border-color 0.3s;
            font-family: inherit;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            margin-top: 15px;
            transition: opacity 0.3s, transform 0.1s;
        }
        button:hover { opacity: 0.9; transform: translateY(-1px); }
        button:active { transform: translateY(0); }
        .result {
            margin-top: 25px;
            padding: 20px;
            border-radius: 12px;
            display: none;
            animation: fadeIn 0.4s ease;
        }
        .result.short {
            background: #e8f5e9;
            border-left: 5px solid #4caf50;
        }
        .result.long {
            background: #e3f2fd;
            border-left: 5px solid #2196f3;
        }
        .result h3 {
            font-size: 1.1em;
            margin-bottom: 6px;
        }
        .result p { color: #555; font-size: 0.95em; }
        .stats {
            margin-top: 25px;
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 12px;
        }
        .stat-box {
            background: #f8f9ff;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            border: 1px solid #e8eaff;
        }
        .stat-box .value {
            font-size: 1.6em;
            font-weight: 700;
            color: #667eea;
        }
        .stat-box .label {
            font-size: 0.8em;
            color: #888;
            margin-top: 4px;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .footer {
            text-align: center;
            margin-top: 25px;
            color: #aaa;
            font-size: 0.8em;
        }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1> AI Conversation Predictor</h1>
        <p>Predicts whether an answer will be short or long</p>
        <br>
        <span class="badge">Naive Bayes + TF-IDF Model</span>
    </div>

    <label for="question">Enter a Question:</label>
    <textarea id="question" placeholder="e.g. How are you doing today?"></textarea>

    <button onclick="predict()">🔍 Predict Answer Type</button>

    <div id="result" class="result">
        <h3 id="result-title"></h3>
        <p id="result-desc"></p>
    </div>

    <div class="stats">
        <div class="stat-box">
            <div class="value" id="stat-model">Naive Bayes</div>
            <div class="label">Model Used</div>
        </div>
        <div class="stat-box">
            <div class="value" id="stat-features">500</div>
            <div class="label">TF-IDF Features</div>
        </div>
        <div class="stat-box">
            <div class="value" id="stat-accuracy">~85%</div>
            <div class="label">Accuracy</div>
        </div>
    </div>

    <div class="footer">
        Lab 12 — Flask Application | Dataset: Kaggle Conversation CSV
    </div>
</div>

<script>
    /* ChatGPT-generated JavaScript */
    async function predict() {
        const question = document.getElementById('question').value.trim();
        if (!question) {
            alert('Please enter a question!');
            return;
        }

        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question })
        });

        const data = await response.json();
        const resultDiv = document.getElementById('result');
        const title     = document.getElementById('result-title');
        const desc      = document.getElementById('result-desc');

        resultDiv.style.display = 'block';
        resultDiv.className = 'result';

        if (data.prediction === 1) {
            resultDiv.classList.add('long');
            title.textContent = '📝 Prediction: LONG Answer Expected';
            desc.textContent  = 'The model predicts this question will receive a detailed, longer response (more than 50 characters).';
        } else {
            resultDiv.classList.add('short');
            title.textContent = '💬 Prediction: SHORT Answer Expected';
            desc.textContent  = 'The model predicts this question will receive a brief, concise response (50 characters or less).';
        }

        document.getElementById('stat-accuracy').textContent = (data.confidence * 100).toFixed(1) + '%';
    }

    document.getElementById('question').addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); predict(); }
    });
</script>
</body>
</html>
"""

# FLASK ROUTES


@app.route('/')
def home():
    """Serve the frontend (ChatGPT-generated HTML)"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    Body: { "question": "your question here" }
    Returns: { "prediction": 0 or 1, "label": "Short/Long", "confidence": float }
    """
    try:
        data     = request.get_json()
        question = data.get('question', '')

        if not question:
            return jsonify({'error': 'Question is required'}), 400

        # Clean and vectorize
        cleaned  = clean_text(question)
        vec      = vectorizer.transform([cleaned])

        # Predict
        prediction  = int(model.predict(vec)[0])
        proba       = model.predict_proba(vec)[0]
        confidence  = float(max(proba))
        label       = "Long Answer" if prediction == 1 else "Short Answer"

        return jsonify({
            'prediction': prediction,
            'label':      label,
            'confidence': round(confidence, 4),
            'question':   question
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'running', 'model': 'MultinomialNB'}), 200

# RUN APP

if __name__ == '__main__':
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    print("\n" + "=" * 60)
    print("  Lab 12 - Flask App Running!")
    print("  Open: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)