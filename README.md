# Heart Disease Prediction

A small Flask web application demonstrating a Random Forest classifier trained on the Cleveland heart-disease dataset. It provides informational pages, a form for a model estimate, and a downloadable PDF summary.

> **Medical disclaimer:** This is an educational project, not a medical device. Its output is not a diagnosis, medical advice, or a substitute for professional care. Do not use it to make health decisions.

## Repository structure

```text
.
├── app.py                         # Flask web application
├── prediction.py                  # Model training and evaluation script
├── Heart_disease_cleveland_new.csv # Dataset used by the training script
├── requirements.txt               # Python dependencies
├── Procfile                       # Gunicorn start command
├── static/                        # CSS and image assets
└── templates/                     # Jinja HTML templates
```

Training creates `heart_disease_model.pkl` and `scaler.pkl` in the project root. They are ignored by Git and must be generated locally (or separately provided) before prediction requests can work.

## Setup

Use Python 3.9 or later and create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the model

From the repository root, run:

```bash
python prediction.py
```

The script reads `Heart_disease_cleveland_new.csv`, checks for the 13 feature columns plus `target`, reports holdout metrics, then writes `heart_disease_model.pkl` and `scaler.pkl`.

## Run the application

```bash
python app.py
```

Open <http://127.0.0.1:5000>. The app's home and information pages can load without model files; prediction requests require successful model training first.

For deployment with Gunicorn, set a strong `SECRET_KEY` environment variable and use:

```bash
gunicorn app:app
```

Do not use the development server or a fallback secret in production.

## Notes and limitations

- The dataset is committed as supplied; verify its source, terms of use, feature encodings, and data quality before reuse.
- The displayed class/probability is a model output, not a clinically validated risk estimate. A small holdout score does not demonstrate clinical usefulness.
- Input ranges and category labels should be checked against the exact dataset encoding before using the form. The UI currently needs review to ensure its descriptions match the dataset's numeric values.
- No automated tests or license file are included yet. Add tests for training, form validation, routes, and PDF generation, and add an appropriate license if you intend to distribute the project.
