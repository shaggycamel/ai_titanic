import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Load datasets
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

def preprocess_data(train_df, test_df):
    # Create copies
    train = train_df.copy()
    test = test_df.copy()
    
    # Keep PassengerId for submission
    test_ids = test['PassengerId'].copy()
    
    # 1. Feature Engineering: Extract Titles from Name
    def extract_title(name):
        if pd.isna(name):
            return 'Unknown'
        return name.split(',')[0].split('.')[1] if '.' in name.split(',')[0] else 'Unknown'
    
    # A bit more robust title extraction
    def get_title(name):
        title = ''
        if ',' in name:
            name = name.split(',')[0]
            if ' ' in name:
                parts = name.split(' ')
                title = parts[1]
            else:
                title = 'Unknown'
        return title

    # Actually, let's use a simpler approach with regex or splitting
    # Name format: "Braund, Mr. Owen Harris"
    train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Group rare titles
    title_map = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Sir': 'Rare', 'Jon': 'Rare', 'Don': 'Rare', 'Lady': 'Rare', 'Count': 'Rare'
    }
    train['Title'] = train['Title'].map(title_map).fillna('Rare')
    test['Title'] = test['Title'].map(title_map).fillna('Rare')

    # 2. Feature Engineering: Family Size
    train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
    test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
    
    train['IsAlone'] = (train['FamilySize'] == 1).astype(int)
    test['IsAlone'] = (test['FamilySize'] == 1).astype(int)

    # 3. Feature Selection: Dropping high-cardinality or useless columns
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    train = train.drop(columns=[c for c in cols_to_drop if c in train.columns])
    test = test.drop(columns=[c for c in cols_to_drop if c in test.columns])
    
    # 4. Handling Missing Values
    # Age: Impute with median
    imputer_age = SimpleImputer(strategy='median')
    train['Age'] = imputer_age.fit_transform(train[['Age']])
    test['Age'] = imputer_age.transform(test[['Age']])
    
    # Embarked: Impute with mode
    imputer_embarked = SimpleImputer(strategy='most_frequent')
    train['Embarked'] = imputer_embarked.fit_transform(train[['Embarked']]).ravel()
    test['Embarked'] = imputer_embarked.transform(test[['Embarked']]).ravel()
    
    # Fare: Impute with median
    imputer_fare = SimpleImputer(strategy='median')
    train['Fare'] = imputer_fare.fit_transform(train[['Fare']])
    test['Fare'] = imputer_fare.transform(test[['Fare']])
    
    # 5. Encoding Categorical Variables
    le_sex = LabelEncoder()
    train['Sex'] = le_sex.fit_transform(train['Sex']).ravel()
    test['Sex'] = le_sex.transform(test['Sex']).ravel()
    
    le_embarked = LabelEncoder()
    train['Embarked'] = le_embarked.fit_transform(train['Embarked']).ravel()
    test['Embarked'] = le_embarked.transform(test['Embarked']).ravel()

    le_title = LabelEncoder()
    train['Title'] = le_title.fit_transform(train['Title']).ravel()
    test['Title'] = le_title.transform(test['Title']).ravel()
    
    return train, test, test_ids

# Preprocess
train_proc, test_proc, test_ids = preprocess_data(train_df, test_df)

# Split for validation
X = train_proc.drop('Survived', axis=1)
y = train_proc['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training & Hyperparameter Tuning
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")

# Evaluate on validation set
best_model = grid_search.best_estimator_
val_predictions = best_model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Predict on test set
test_predictions = best_model.predict(test_proc)

# Prepare submission
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Survived': test_predictions
})

submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")
