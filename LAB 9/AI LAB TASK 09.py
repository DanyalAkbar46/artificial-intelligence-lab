import pandas as pd

df = pd.read_csv("train_and_test2.csv")

print("=== Dataset Shape ===")
print(df.shape)

print("\n=== Column Names ===")
print(df.columns)

print("\n=== Info ===")
print(df.info())

print("\n=== Summary Statistics ===")
print(df.describe())

print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Sex Distribution ===")
print(df['Sex'].value_counts())

print("\n=== Survival Distribution ===")
print(df['2urvived'].value_counts())
