# DopaScore: Predicting Productivity Risk Bands from Digital Behavior Patterns

## Overview
In the modern attention economy, persuasive digital algorithms and variable-reward notification architectures subject human attentional systems to chronic dopaminergic hyper-stimulation, accelerating attention fragmentation, cognitive fatigue, and acute productivity decline. **DopaScore** is an end-to-end predictive machine learning and behavioral analytics solution developed for the **IBM SkillsBuild AICTE "Data Analytics with AI" Internship**. By conducting a rigorous data audit to rectify dataset schema mislabelings and strictly isolating downstream outcome scores to prevent data leakage, DopaScore processes 54 genuine behavioral habits, screen time metrics, focus stamina indicators, and psychological dopamine indicators to diagnose an individual's **Productivity Risk Band** (`None/Minimal`, `Early`, `Moderate`, `Severe`, `Critical`). Achieving over 98% macro F1-score across evaluated ensemble models, DopaScore translates complex non-linear digital behavioral footprints into early-stage diagnostic warnings and actionable digital hygiene protocols.

---

## Dataset Source
- **Kaggle Dataset Source**: [Social Media, Dopamine, and Productivity Dataset](https://www.kaggle.com/datasets/manaswinsripatnala/social-media-dopamine-and-productivity-dataset)  
  *Local File:* `social_media_dopamine_productivity.csv` (300 clean participant records, 66 original columns).  
  *Schema Audit Note:* The original uploader inverted `productivity_decline_score` (which genuinely holds the categorical risk bands) and `severity_stage` (which holds trigger text with missingness). In this project, the target is formally verified and mapped to `productivity_risk_band`.

---

## Quick Start

### 1. Clone or Open the Repository
```bash
git clone https://github.com/suhaniagarwal18/dopascore-productivity-risk-predictor.git
cd dopascore-productivity-risk-predictor
```

### 2. Create and Activate a Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- **Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Installed Packages & Responsibilities:**
- `pandas (>=2.2.0)`, `numpy (>=2.0.0)` — Data ingestion, cleaning, transformation, and array operations
- `scikit-learn (>=1.5.0)` — Preprocessing pipelines (`ColumnTransformer`, `StandardScaler`, encoders) and classification models
- `scipy (>=1.14.0)` — Scientific computing routines and statistical distributions
- `matplotlib (>=3.9.0)`, `seaborn (>=0.13.0)` — Exploratory data analysis, heatmaps, and confusion matrix plots
- `joblib (>=1.4.0)` — Serialization and persistence of the fitted model pipeline
- `streamlit (>=1.38.0)` — Interactive web application UI, state management, and real-time inference
- `python-docx (>=1.1.0)` — Programmatic generation and formatting of the project report
- `nbformat (>=5.10.0)`, `nbclient (>=0.11.0)`, `ipykernel (>=6.29.0)` — Jupyter notebook execution, validation, and kernel runtime

---

## Train the Model
To execute the complete exploratory data analysis, run the column verification audit, benchmark the models, and regenerate the serialized pipeline artifact (`dopascore_pipeline.joblib`), run the Jupyter notebook:

### Option A: Using Jupyter Lab / Notebook Interface
```bash
jupyter notebook Suhani_DopaScore.ipynb
```
*Click **Kernel** > **Restart & Run All Cells**.*

### Option B: Headless Command-Line Execution
```bash
# Using python nbclient in your environment
python -c "import nbformat; from nbclient import NotebookClient; nb = nbformat.read('Suhani_DopaScore.ipynb', as_version=4); NotebookClient(nb, timeout=600).execute(); nbformat.write(nb, open('Suhani_DopaScore.ipynb', 'w', encoding='utf-8'))"
```
*Execution generates and saves the production pipeline `dopascore_pipeline.joblib`.*

---

## Run the App Locally
The project features a single-file, self-contained Streamlit application (`app.py`) integrating both the interactive frontend UI and the machine learning inference engine.

Launch the local web application with:
```bash
streamlit run app.py
```

Once started, the application will automatically open in your default browser at:
- **Local URL:** `http://localhost:8501`

Use the interactive sliders to adjust your daily digital consumption, focus metrics, and dopamine triggers to receive an instant productivity risk diagnosis with colored visual status badges, probability breakdowns, and personalized digital hygiene recommendations.

---

## Technologies Used
- **Programming Language:** Python 3.11+
- **Data Manipulation & Processing:** `pandas` (>=2.2.0), `numpy` (>=2.0.0), `scipy` (>=1.14.0)
- **Machine Learning & Preprocessing:** `scikit-learn` (>=1.5.0) (`ColumnTransformer`, `StandardScaler`, `OrdinalEncoder`, `OneHotEncoder`, `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`)
- **Model Persistence & Pipeline Serialization:** `joblib` (>=1.4.0)
- **Data Visualization & Analytics:** `matplotlib` (>=3.9.0), `seaborn` (>=0.13.0)
- **Interactive Web Application:** `streamlit` (>=1.38.0)
- **Formal Documentation & Reporting:** `python-docx` (>=1.1.0), `nbformat` (>=5.10.0), `nbclient` (>=0.11.0)

---

## Project Structure
```text
dopascore-productivity-risk-predictor
├── app.py                                   # Standalone single-file Streamlit web application
├── dopascore_pipeline.joblib                # Serialized trained ColumnTransformer + Random Forest pipeline
├── README.md                                # Comprehensive project documentation & guide
├── requirements.txt                         # Pinned Python package dependencies
├── social_media_dopamine_productivity.csv   # Raw benchmark dataset (300 valid participant records)
├── Suhani_DopaScore.ipynb                   # End-to-end Jupyter analytics & modeling notebook
└── Suhani_ProjectReport.docx                # Professional formal Word project report
```
