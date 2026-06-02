import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Display basic information about the dataset
print("Train dataset shape:", train_df.shape)
print("Test dataset shape:", test_df.shape)
print("\nTrain dataset info:")
print(train_df.info())
print("\nFirst few rows of train dataset:")
print(train_df.head())

# Check for missing values
print("\nMissing values in train dataset:")
print(train_df.isnull().sum())
print("\nMissing values in test dataset:")
print(test_df.isnull().sum())

# Data preprocessing
# Fill missing ages with median
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())

# Fill missing embarked with mode
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

# Fill missing fare with median
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

# Create a new feature: FamilySize
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1

# Encode categorical variables
le = LabelEncoder()
train_df['Sex'] = le.fit_transform(train_df['Sex'])
test_df['Sex'] = le.transform(test_df['Sex'])

# Encode embarked
train_df['Embarked'] = le.fit_transform(train_df['Embarked'])
test_df['Embarked'] = le.transform(test_df['Embarked'])

# Select features for training
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize']
X = train_df[features]
y = train_df['Survived']

# Split data for training
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"\nModel accuracy: {accuracy:.4f}")

# Make predictions on test set
test_predictions = model.predict(test_df[features])
test_df['Survived'] = test_predictions

# Prepare submission file
submission = test_df[['PassengerId', 'Survived']]
submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")