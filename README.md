# AG News Classification — Deep Learning Project

> **Course:** DL MATH — Semester 2  
> **Objective:** Build, train, and evaluate deep-learning models (LSTM from scratch + BERT) on the [AG News Classification Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) to categorize news articles into **World, Sports, Business,** and **Sci/Tech**.

---

## Business Context

A mid-sized media-monitoring firm processes **~50,000 news articles per day** manually.  
Automating news triage with a **94 % accurate classifier** reduces human curation effort by **~80 %**, translating to an estimated **cost saving of $930,000 / year** (assuming $25/hr analyst wages and 2 min/article review time).

---

## Project Structure (Lean — One Notebook Workflow)

```
├── data/                  # Raw & processed datasets (git-ignored)
├── src/                   # Reusable Python modules
│   ├── preprocessing.py   # Text cleaning, tokenization, padding
│   ├── model_builder.py   # Compiled Keras models (LSTM & BERT)
│   └── utils.py           # Plotting & evaluation helpers
├── notebooks/
│   └── dl_math_project.ipynb   # ← SINGLE main notebook
├── models/                # Saved weights & tokenizers (git-ignored)
├── app/
│   └── app.py             # Streamlit demo
├── reports/               # Generated plots & final PDF
├── requirements.txt
├── setup.sh / setup.bat
└── README.md
```

**Philosophy:**  
All exploratory analysis, preprocessing, training, evaluation, and visualisation live in **one** notebook (`notebooks/dl_math_project.ipynb`).  The `src/` package keeps the notebook clean by hiding boiler-plate code; the notebook simply imports and calls well-documented functions.

---

## Quick Start

### 1. Setup Environment

**Linux / macOS:**
```bash
bash setup.sh
```

**Windows:**
```cmd
setup.bat
```

Or manually:
```bash
python -m venv venv
# Activate (see OS-specific commands below)
pip install -r requirements.txt
```

### 2. Run the Project

1. Download the AG News dataset from [Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) and place `train.csv` + `test.csv` inside `data/`.
2. Open `notebooks/dl_math_project.ipynb` in Jupyter or VS Code.
3. **Run all cells top-to-bottom** — this executes:
   - Data loading & inspection
   - Preprocessing (`src/preprocessing`)
   - Model building (`src/model_builder`)
   - Training & fine-tuning
   - Evaluation, plots, and report generation (`src/utils`)
   - Saving artifacts to `models/` and `reports/`

> **Tip:** If you only want to see inference results without retraining, load the saved model from `models/` using the paths printed in the notebook.

### 3. Launch the Streamlit Demo (separate terminal)

```bash
# Activate your virtual environment first
streamlit run app/app.py
```

The demo will open in your browser and let you type custom news headlines for real-time classification.

---

## Dataset

- **Source:** [AG News Classification Dataset on Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
- **Classes:** 4 — World, Sports, Business, Sci/Tech
- **Splits:** 120,000 training samples, 7,600 test samples
- **Columns:** `Class Index`, `Title`, `Description`

---

## Requirements

Python 3.9+ is recommended. See `requirements.txt` for the full dependency list.

---

## License

This project is for educational purposes only.
