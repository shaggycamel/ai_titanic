#!/usr/bin/env python3
"""
Titanic ML Pipeline
Implements a complete ML pipeline: load → clean → feature engineer → train → evaluate → output
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data():
    """Load and preprocess training and test datasets."""
    print("=" * 80)
    print("STEP 1: Loading and Preprocessing Data")
    print("=" * 80)

    # Load data
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    print(f"\nTraining set: {train_df.shape}")
    print(f"Test set: {test_df.shape}")

    # Create copy for modeling
    X_train = train_df.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], axis=1)
    y_train = train_df['Survived']
    X_test = test_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

    print(f"\nFeatures for training: {X_train.shape}")
    print(f"Features for testing: {X_test.shape}")

    return X_train, X_test, y_train


def engineer_features(X_train, X_test):
    """Additional feature engineering."""
    print("\n" + "=" * 80)
    print("STEP 2: Feature Engineering")
    print("=" * 80)

    # Create family size feature
    X_train['FamilySize'] = X_train['SibSp'] + X_train['Parch']
    print("✓ Created FamilySize feature")

    # Keep a copy of original for analysis
    print("\nSample features (after engineering):")
    feature_cols = ['Age', 'Fare', 'Pclass', 'Sex', 'SibSp', 'Parch', 'FamilySize']
    available_cols = [col for col in feature_cols if col in X_train.columns]
    print(X_train[available_cols].head())

    return X_train, X_test


def build_models():
    """Build multiple candidate models."""
    print("\n" + "=" * 80)
    print("STEP 3: Building Multiple Candidate Models")
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


def train_and_evaluate(X_train, y_train, X_test, models, preprocessor):
    """Train and evaluate models."""
    print("\n" + "=" * 80)
    print("STEP 4: Model Training and Evaluation")
    print("=" * 80)

    results = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        try:
            # Train
            pipeline.fit(X_train, y_train)
            print(f"✓ {model_name} trained successfully")

            # Cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
            print(f"  Cross-validation accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

            # Evaluate on train set for validation
            y_train_pred = pipeline.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            print(f"  Training accuracy: {train_accuracy:.4f}")

            # Evaluate on test set
            y_pred = pipeline.predict(X_test)

            # Create submission for evaluation
            submission = pd.DataFrame({
                'PassengerId': pd.read_csv('data/test.csv')['PassengerId'],
                'Survived': y_pred
            })

            results[model_name] = {
                'pipeline': pipeline,
                'cv_accuracy': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_accuracy': train_accuracy,
                'y_pred': y_pred
            }

        except Exception as e:
            print(f"  ❌ Error training {model_name}: {e}")
            continue

    # Initialize submission with empty values if results is empty
    submission = pd.DataFrame({
        'PassengerId': pd.read_csv('data/test.csv')['PassengerId'],
        'Survived': []
    })

    return results, submission


def select_best_model(results):
    """Select the best performing model."""
    print("\n" + "=" * 80)
    print("STEP 5: Model Comparison and Selection")
    print("=" * 80)

    # Sort models by cross-validation accuracy
    sorted_models = sorted(results.items(), key=lambda x: x[1]['cv_accuracy'], reverse=True)

    print("\nModel Performance (Cross-Validation Accuracy):")
    print("-" * 60)
    for i, (model_name, result) in enumerate(sorted_models, 1):
        print(f"{i}. {model_name}: {result['cv_accuracy']:.4f} (±{result['cv_std']:.4f})")

    # Select best model
    best_model_name, best_model = sorted_models[0]
    print(f"\n✓ Selected model: {best_model_name}")
    print(f"  Performance: {best_model_name} with {best_model['cv_accuracy']:.4f} average accuracy")

    return best_model, best_model_name


def generate_submission(best_model, best_model_name, X_test, final_submission=False):
    """Generate final submission file."""
    print("\n" + "=" * 80)
    print("STEP 6: Final Submission Generation")
    print("=" * 80)

    y_pred = best_model['y_pred']
    passenger_ids = pd.read_csv('data/test.csv')['PassengerId']

    # Create submission
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': y_pred
    })

    print(f"\nSubmission shape: {submission.shape}")
    print(f"Survived count: {submission['Survived'].sum()}")
    print(f"Deceased count: {submission['Survived'].sum(0)}")

    # Save to file
    if final_submission:
        output_filename = 'submission_titanic.csv'
        submission.to_csv(output_filename, index=False)
        print(f"✓ Final submission saved to: {output_filename}")
    else:
        output_filename = 'prediction_titanic.csv'
        submission.to_csv(output_filename, index=False)
        print(f"✓ Predictions saved to: {output_filename}")

    return submission


def main():
    """Main execution function."""
    try:
        # Load and preprocess data
        X_train, X_test, y_train = load_and_preprocess_data()

        # Feature engineering
        X_train, X_test = engineer_features(X_train, X_test)

        print("\n" + "=" * 80)
        print("STEP 3: Building Preprocessing Pipeline")
        print("=" * 80)

        # Define features explicitly
        numerical_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
        categorical_features = ['Pclass', 'Sex', 'Embarked']

        from sklearn.impute import SimpleImputer

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        print(f"Numerical features: {numerical_features}")
        print(f"Categorical features: {categorical_features}")

        # Build models
        models = build_models()

        # Train and evaluate
        results, submission = train_and_evaluate(X_train, y_train, X_test, models, preprocessor)

        # Select best model
        if results:
            best_model, best_model_name = select_best_model(results)

            # Generate final submission
            final_submission = generate_submission(best_model, best_model_name, X_test, final_submission=True)

            print("\n" + "=" * 80)
            print("ML PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"\nBest model: {best_model_name}")
            print(f"Submission file: submission_titanic.csv")
            print(f"Predicted survivors: {final_submission['Survived'].sum()}/{len(final_submission)}")

            return final_submission
        else:
            print("\n❌ No models were successfully trained")
            return None

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()