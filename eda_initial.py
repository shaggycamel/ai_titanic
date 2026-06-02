import pandas as pd

# Load datasets
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Display basic info
print("--- Train Info ---")
print(train_df.info())
print("\n--- Train Head ---")
print(train_df.head())
print("\n--- Train Summary Statistics ---")
print(train_df.describe())
print("\n--- Test Info ---")
print(test_df.info())
print("\n--- Missing Values in Train ---")
print(train_df.isnull().sum())
print("\n--- Missing Values in Test ---")
print(test_df.isnull().sum())
