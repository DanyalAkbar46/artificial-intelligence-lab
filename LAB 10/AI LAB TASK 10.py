import pandas as pd

df = pd.read_csv("titanic.csv")

print("Shape:", df.shape)
print("Columns:", df.columns)
print(df.info())
print(df.describe())
print("Missing values:\n", df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop(columns=['Cabin'])

df['Survived'] = df['Survived'].astype(int)
df['Pclass'] = df['Pclass'].astype(int)
df['Age'] = df['Age'].astype(int)

print(df.info())
print(df.head())
