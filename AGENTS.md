# OpenCode Agent Quick‑Start Guide

## Target: Reduce agent confusion

The following points capture every command or convention that an agent would otherwise guess incorrectly for this repo.

- **Run the app** – `python main.py` (or `./main.py` if executable). No CLI arguments.
- **Project task** – Train a model on `data/train.csv` (contains `Survived`) to predict survival of passengers in `data/test.csv`.
- **Dataset location** – All data files live in `data/`. `train.csv` houses the label, `test.csv` does not.
- **Virtual environment** – The repo includes a `.venv`. Activate before running agents that import packages: `source .venv/bin/activate`.
- **Package installation** – Use `uv add <package>` if additional packages are needed.
- **Linting and code format** – Ruff is configured via `ruff.toml`. Use `ruff check .` to validate lint rules.
- **Single‑package layout** – All code resides in the `src/ai_titanic` directory (currently empty), with main.py in root

- **Custom agents** – Place YAML files under `agents/` with a matching Python module. `opencode run --agent <name>` will discover them.
- **Testing not provided** – No test suite exists.
- **No build artifacts** – Nothing is compiled or generated.

### Quick checks before running agents
- `python --version` should be ≥ 3.12.
- `pip list | grep pandas` should show pandas ≥ 3.0.3 (from pyproject).
- `python -m opencode.version` prints the OpenCode version.


## Overarching Goal
Build a full ML pipeline: load → clean → feature engineer → train → evaluate → output

## What you need to do

### Data

You’ll gain access to two similar datasets that include passenger information like name, age, gender, socio-economic class, etc. One dataset is titled train.csv and the other is titled test.csv.

Train.csv will contain the details of a subset of the passengers on board (891 to be exact) and importantly, will reveal whether they survived or not, also known as the “ground truth”.

The test.csv dataset contains similar information but does not disclose the “ground truth” for each passenger. It’s your job to predict these outcomes.

Using the patterns you find in the train.csv data, predict whether the other 418 passengers on board (found in test.csv) survived.

### Job

It is your job to predict if a passenger survived the sinking of the Titanic or not.
For each in the test set, you must predict a 0 or 1 value for the variable.

This will require carrying out exploratory analysis and summary statistics, understanding interactions between variables, comprehensive feature engineering and accurate model selection and tuning.

### Metric

Your score is the percentage of passengers you correctly predict. This is known as accuracy.

### Submission File Format

You should produce a csv file with exactly 418 entries plus a header row. Your submission will show an error if you have extra columns (beyond PassengerId and Survived) or rows.

The file should have exactly 2 columns:
 - PassengerId (sorted in any order)
 - Survived (contains your binary predictions: 1 for survived, 0 for deceased)