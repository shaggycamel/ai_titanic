#!/usr/bin/env python3
"""
Titanic Dataset Analysis and Data Cleaning Script

This script performs:
1. Loading and exploration of the Titanic dataset
2. Summary statistics and exploratory analysis
3. Data cleaning and preprocessing
4. Analysis of key trends and insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def load_data():
    """Load the Titanic train and test datasets."""
    print("=" * 80)
    print("STEP 1: Loading Data")
    print("=" * 80)

    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    print(f"\nTraining set shape: {train_df.shape}")
    print(f"Testing set shape: {test_df.shape}\n")

    return train_df, test_df


def explore_data(train_df, test_df):
    """Perform exploratory data analysis and show summary statistics."""
    print("=" * 80)
    print("STEP 2: Exploratory Data Analysis & Summary Statistics")
    print("=" * 80)

    # Dataset overview
    print("\nTRAINING SET OVERVIEW:")
    print(train_df.info())
    print("\n" + "-" * 80)

    # Summary statistics for numerical columns
    print("\nSUMMARY STATISTICS (NUMERICAL COLUMNS):")
    print(train_df.describe())
    print("\n" + "-" * 80)

    # Summary statistics for categorical columns
    print("\nSUMMARY STATISTICS (CATEGORICAL COLUMNS):")
    print(train_df.describe(include=['object']))
    print("\n" + "-" * 80)

    # Survival distribution
    print("\nSURVIVAL DISTRIBUTION:")
    survival_counts = train_df['Survived'].value_counts()
    survival_percent = train_df['Survived'].value_counts(normalize=True) * 100
    print(pd.DataFrame({
        'Count': survival_counts,
        'Percentage': survival_percent.round(2)
    }))
    print("\n" + "-" * 80)

    # Missing values analysis
    print("\nMISSING VALUES ANALYSIS:")
    print(train_df.isnull().sum())
    print("\n" + "-" * 80)

    # Data type analysis
    print("\nDATA TYPE ANALYSIS:")
    print(pd.DataFrame({
        'Column': train_df.columns,
        'Dtype': train_df.dtypes.values
    }))
    print("\n" + "-" * 80)

    return train_df, test_df


def clean_data(train_df, test_df):
    """Clean and preprocess the data."""
    print("\n" + "=" * 80)
    print("STEP 3: Data Cleaning")
    print("=" * 80)

    # Create a copy to avoid modifying original data
    train_clean = train_df.copy()
    test_clean = test_df.copy()

    print(f"\nInitial missing values in train set:")
    print(train_clean.isnull().sum())

    # Handle missing values in training set
    print("\n" + "-" * 80)
    print("Handling missing values in training set:")

    # Age: Fill with median (common practice)
    train_clean['Age'] = train_clean['Age'].fillna(train_clean['Age'].median())
    print(f"✓ Age: Filled {train_clean['Age'].isnull().sum()} null values with median {train_clean['Age'].median():.2f}")

    # Embarked: Fill with most common port
    train_clean['Embarked'] = train_clean['Embarked'].fillna(train_clean['Embarked'].mode()[0])
    print(f"✓ Embarked: Filled {train_clean['Embarked'].isnull().sum()} null values with mode {train_clean['Embarked'].mode()[0]}")

    # Test set missing values
    print("\n" + "-" * 80)
    print("Handling missing values in test set:")

    # Age: Fill with median
    test_clean['Age'] = test_clean['Age'].fillna(test_clean['Age'].median())
    print(f"✓ Age: Filled {test_clean['Age'].isnull().sum()} null values with median {test_clean['Age'].median():.2f}")

    # Fare: Fill with median
    test_clean['Fare'] = test_clean['Fare'].fillna(test_clean['Fare'].median())
    print(f"✓ Fare: Filled {test_clean['Fare'].isnull().sum()} null values with median {test_clean['Fare'].median():.2f}")

    # Cabin: Drop column due to many missing values
    train_clean = train_clean.drop('Cabin', axis=1)
    test_clean = test_clean.drop('Cabin', axis=1)
    print(f"✓ Cabin: Dropped column due to {train_df['Cabin'].isnull().sum()} missing values")

    print("\n" + "-" * 80)
    print("Final missing values in cleaned datasets:")
    print("\nTraining set:")
    print(train_clean.isnull().sum())
    print("\nTest set:")
    print(test_clean.isnull().sum())

    # Categorical encoding for training set
    print("\n" + "-" * 80)
    print("Categorical encoding for training set:")
    train_clean_dummies = pd.get_dummies(train_clean, columns=['Sex', 'Embarked'], drop_first=True)
    print(f"✓ Applied one-hot encoding to Sex and Embarked")

    # Categorical encoding for test set
    test_clean_dummies = pd.get_dummies(test_clean, columns=['Sex', 'Embarked'], drop_first=True)
    print(f"✓ Applied one-hot encoding to Sex and Embarked")

    return train_clean_dummies, test_clean_dummies


def analyze_trends(train_clean):
    """Analyze key trends and insights from the dataset."""
    print("\n" + "=" * 80)
    print("STEP 4: Trend Analysis & Key Insights")
    print("=" * 80)

    # Survival by Gender
    print("\n" + "-" * 80)
    print("1. SURVIVAL BY GENDER:")

    # Get columns that start with 'Sex' or contains 'fe' (female)
    sex_cols = [col for col in train_clean.columns if 'Sex' in col or 'fe' in col]
    print(f"Sex-related columns after encoding: {sex_cols}")

    # Handle different encoding scenarios
    if 'Sex_male' in train_clean.columns:
        # 1 = male, 0 = female
        female_survival = train_clean[train_clean['Sex_male'] == 0]['Survived'].mean()
        male_survival = train_clean[train_clean['Sex_male'] == 1]['Survived'].mean()
        gender_survival = pd.DataFrame({
            'Female (0)': female_survival,
            'Male (1)': male_survival
        }, index=['Survival Rate'])
        print(gender_survival)
    elif 'Sex_female' in train_clean.columns:
        # Similar handling if column name is different
        female_survival = train_clean[train_clean['Sex_female'] == 1]['Survived'].mean()
        male_survival = train_clean[train_clean['Sex_female'] == 0]['Survived'].mean()
        gender_survival = pd.DataFrame({
            'Female (1)': female_survival,
            'Male (0)': male_survival
        }, index=['Survival Rate'])
        print(gender_survival)
    else:
        # Fallback: use original 'Sex' column
        gender_survival = train_clean.groupby('Sex')['Survived'].agg(['mean', 'count'])
        gender_survival.columns = ['Survival Rate', 'Passenger Count']
        print(gender_survival)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


def main():
    """Main execution function."""
    try:
        # Load data
        train_df, test_df = load_data()

        # Explore data
        train_df, test_df = explore_data(train_df, test_df)

        # Clean data
        train_clean, test_clean = clean_data(train_df, test_df)

        # Analyze trends
        analyze_trends(train_clean)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 80)

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found - {e}")
        print("Please ensure data files are in the 'data/' directory.")
    except Exception as e:
        print(f"\n❌ ERROR: An unexpected error occurred - {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()