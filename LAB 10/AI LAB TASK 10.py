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
import pandas as pd

# Fill missing Age with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing Embarked with mode (most frequent value)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin column if present (too many missing values)
if 'Cabin' in df.columns:
    df = df.drop(columns=['Cabin'])

# Convert datatypes to integer where appropriate
df['Age'] = df['Age'].astype(int)
df['Pclass'] = df['Pclass'].astype(int)
df['sibsp'] = df['sibsp'].astype(int)
df['Parch'] = df['Parch'].astype(int)
df['Sex'] = df['Sex'].astype(int)
df['2urvived'] = df['2urvived'].astype(int)

print("=== After Preprocessing ===")
print(df.info())
print(df.head())
)
