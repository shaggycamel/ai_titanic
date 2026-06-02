#!/usr/bin/env python3
"""
Titanic ML Pipeline - Working Version
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data():
    """Load and preprocess data."""
    print("=" * 80)
    print("STEP 1: Loading Data")
    print("=" * 80)

    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    print(f"\nTraining set: {train_df.shape}")
    print(f"Test set: {test_df.shape}")

    X_train = train_df.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], axis=1)
    y_train = train_df['Survived']
    X_test = test_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

    print(f"\nFeatures for training: {X_train.shape}")
    print(f"Features for testing: {X_test.shape}")

    return X_train, X_test, y_train


def engineer_features(X_train, X_test):
    """Feature engineering."""
    print("\n" + "=" * 80)
    print("STEP 2: Feature Engineering")
    print("=" * 80)

    X_train['FamilySize'] = X_train['SibSp'] + X_train['Parch']
    X_test['FamilySize'] = X_test['SibSp'] + X_test['Parch']
    print("✓ Created FamilySize feature")

    print("\nSample features (after engineering):")
    feature_cols = ['Age', 'Fare', 'Pclass', 'Sex', 'SibSp', 'Parch', 'FamilySize']
    available_cols = [col for col in feature_cols if col in X_train.columns]
    print(X_train[available_cols].head())

    return X_train, X_test


def build_models():
    """Build candidate models."""
    print("\n" + "=" * 80)
    print("STEP 3: Building Models")
    print("=" * 80)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    print("\nAvailable models:")
    for name, model in models.items():
        print(f"  - {name}")

    return models


def train_and_evaluate(X_train, y_train, X_test, models):
    """Train and evaluate models."""
    print("\n" + "=" * 80)
    print("STEP 4: Training Models")
    print("=" * 80)

    results = {}
    passenger_ids = pd.read_csv('data/test.csv')['PassengerId']
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': None
    })

    # Define features
    numerical_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
    categorical_features = ['Pclass', 'Sex', 'Embarked']

    from sklearn.impute import SimpleImputer

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        try:
            pipeline.fit(X_train, y_train)
            print(f"✓ {model_name} trained successfully")

            train_accuracy = accuracy_score(y_train, pipeline.predict(X_train))
            print(f"  Training accuracy: {train_accuracy:.4f}")

            # Get predictions
            y_pred = pipeline.predict(X_test)
            submission['Survived'] = y_pred

            results[model_name] = {
                'pipeline': pipeline,
                'train_accuracy': train_accuracy,
                'y_pred': y_pred
            }

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results, submission


def select_best_model(results):
    """Select best model."""
    if not results:
        return None, None

    sorted_models = sorted(results.items(), key=lambda x: x[1]['train_accuracy'], reverse=True)
    best_model_name, best_model = sorted_models[0]
    print(f"\n✓ Selected model: {best_model_name}")

    return best_model['pipeline'], best_model_name


def generate_submission(best_model, best_model_name, X_test):
    """Generate submission file."""
    print("\n" + "=" * 80)
    print("STEP 5: Generating Submission")
    print("=" * 80)

    y_pred = best_model.predict(X_test)

    submission = pd.DataFrame({
        'PassengerId': pd.read_csv('data/test.csv')['PassengerId'],
        'Survived': y_pred
    })

    submission.to_csv('submission.csv', index=False)
    print(f"✓ Predictions saved to: submission.csv")
    print(f"  Total predictions: {len(submission)}")
    print(f"  Predicted survivors: {submission['Survived'].sum()}/{len(submission)}")

    return submission


def main():
    """Main execution."""
    try:
        # Load and preprocess data
        X_train, X_test, y_train = load_and_preprocess_data()

        # Feature engineering
        X_train, X_test = engineer_features(X_train, X_test)

        # Build models
        models = build_models()

        # Train and evaluate
        results, submission = train_and_evaluate(X_train, y_train, X_test, models)

        if results:
            best_model, best_model_name = select_best_model(results)
            final_submission = generate_submission(best_model, best_model_name, X_test)

            print("\n" + "=" * 80)
            print("ML PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"\nBest model: {best_model_name}")
            print(f"Submission file: submission.csv")
            print(f"Predicted survivors: {final_submission['Survived'].sum()}/{len(final_submission)}")

            return final_submission
        else:
            print("\n❌ No models trained successfully")
            return None

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()