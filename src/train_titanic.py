"""
Titanic survival prediction pipeline.

This script performs the following steps:
1. Loads the official Kaggle Titanic data.
2. Conducts minimal exploratory analysis and feature engineering.
3. Trains several models (LogisticRegression, RandomForest, ExtraTrees, XGBClassifier).
4. Selects the best performing model via cross‑validation.
5. Retrains the chosen model on the full training set.
6. Predicts survival probabilities for the test set.
7. Outputs a CSV file `pred.csv` with the required columns.

The pipeline is intentionally simple but modular and can be extended.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score

# XGBoost may not be available in the base environment – import lazily
try:
    from xgboost import XGBClassifier
    has_xgb = True
except Exception:  # pragma: no cover
    has_xgb = False

DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
PRED_PATH = Path("pred.csv")


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


# Mapping for rare titles
_RARE_TITLES = {
    "Capt": "Other",
    "Col": "Other",
    "Don": "Other",
    "Dr": "Other",
    "Jonkheer": "Other",
    "Major": "Other",
    "Rev": "Other",
    "Sir": "Other",
    "Lady": "Other",
    "the Countess": "Other",
}


def preprocess(df: pd.DataFrame, is_train: bool = True):
    """Return features and optional target.
    Missing values are imputed, categorical variables one‑hot
    encoded, and numeric features scaled.
    """
    df = df.copy()
    # Basic engineered features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    # Title extraction
    df["Title"] = df["Name"].str.extract(r",\s*(\w+)\. ")
    df["Title"].fillna("Other", inplace=True)
    # Reduce rare titles to Other and consolidate Miss/Mlle to Miss
    df.loc[df["Title"].isin(["Mlle", "Ms"]), "Title"] = "Miss"
    df["Title"].replace(_RARE_TITLES, inplace=True)
    # Deck from Cabin
    df["Deck"] = df["Cabin"].str[0]
    df["Deck"].fillna("Unknown", inplace=True)
    # Ticket prefix
    df["TicketPrefix"] = df["Ticket"].str.split().str[0]
    df["TicketPrefix"].fillna("None", inplace=True)

    y = None
    if is_train:
        y = df["Survived"]
        df.drop(columns=["Survived"], inplace=True)

    # Drop columns that add little value or are unnecessary
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    df.drop(columns=drop_cols, inplace=True)

    # Define numeric and categorical columns
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=[object, "category"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    if is_train:
        return df, y, preprocessor
    else:
        return df, preprocessor



def train_and_evaluate(X, y, preprocessor):
    """Return best trained model and its accuracy.
    Uses train/validation split with stratification.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = [
        ("logreg", Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])),
        ("rf", Pipeline([("prep", preprocessor), ("clf", RandomForestClassifier(n_estimators=200, random_state=42))])),
        ("et", Pipeline([("prep", preprocessor), ("clf", ExtraTreesClassifier(n_estimators=200, random_state=42))])),
    ]
    if has_xgb:
        models.append(
            (
                "xgb",
                Pipeline([
                    ("prep", preprocessor),
                    ("clf", XGBClassifier(
                        use_label_encoder=False,
                        eval_metric="logloss",
                        n_estimators=400,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                    )),
                ]),
            )
        )

    best_score = 0.0
    best_model = None
    best_name = ""
    for name, model in models:
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        print(f"{name} accuracy: {acc:.4f}")
        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = name
    print(f"Best model: {best_name} with accuracy {best_score:.4f}")
    return best_model



def main():
    train_df, test_df = load_data()
    X_train_raw, y_train, preprocessor = preprocess(train_df, is_train=True)
    best_model = train_and_evaluate(X_train_raw, y_train, preprocessor)

    # Retrain on full training set
    full_model = best_model
    full_model.fit(X_train_raw, y_train)

    X_test_raw, test_preprocessor = preprocess(test_df, is_train=False)
    # Ensure same preprocessing steps
    test_preprocessor = preprocessor
    preds = full_model.predict(X_test_raw)

    output = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": preds})
    output.to_csv(PRED_PATH, index=False)
    print(f"Saved predictions to {PRED_PATH}")


if __name__ == "__main__":
    main()
