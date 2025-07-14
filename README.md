# OKCupid Date-A-Scientist: Drug Use Prediction

## Project Overview

- **Goal:** Predict drug use patterns among OKCupid users using machine learning models.
- **Data:** Utilizes `profiles.csv` dataset with user attributes and essay responses.
- **Approach:**
  - Data cleaning and feature engineering (e.g., education and ethnicity bucketing, diet strictness)
  - Multiclass and binary classification for drug use prediction
  - Models: Decision Tree, Logistic Regression (with custom thresholding), Random Forest
  - Performance metrics: F1 Macro, Recall, Confusion Matrix visualizations
- **Utilities:**
  - All custom preprocessing and thresholding functions in `tools.py`
  - Notebook (`date-a-scientist.ipynb`) contains full workflow, analysis, and plots
  - HTML export available for sharing results

## Repository Structure

- `date-a-scientist.ipynb` — Main analysis notebook
- `tools.py` — Utility functions for feature engineering and thresholding
- `profiles.csv` — Raw dataset
- `date-a-scientist.html` — HTML export of notebook
- `.gitignore` — Ignores cache and temp files

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Run notebook: Open `date-a-scientist.ipynb` in Jupyter or VS Code
3. View results: See plots and metrics in notebook or `date-a-scientist.html`

---

> For technical details, see comments in `tools.py` and cell explanations in the notebook.
