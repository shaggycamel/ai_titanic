# Titanic Survival Prediction

This project predicts whether passengers survived the Titanic disaster using machine learning.

## Project Structure

- `data/` - Directory containing the dataset files
  - `train.csv` - Training dataset with survival labels
  - `test.csv` - Test dataset without survival labels
- `main.py` - Main Python script that runs the analysis and prediction
- `submission.csv` - Final submission file for Kaggle competition
- `requirements.txt` - Python dependencies

## Setup

1. Install Python 3.8 or higher
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install required packages using uv:
   ```bash
   uv pip install -e .
   ```

## Running the Analysis

Execute the main script:
```bash
python main.py
```

This script will:
1. Load and explore the datasets
2. Preprocess the data (handle missing values, encode categorical variables)
3. Train a Random Forest model
4. Evaluate the model's performance
5. Generate predictions for the test set
6. Create a submission file

## Output

The script generates a `submission.csv` file that can be submitted to the Kaggle Titanic competition.