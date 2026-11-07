# Heart Disease Prediction

A Flask web application that uses a scikit-learn Random Forest model to estimate a heart-disease risk category from clinical input fields. It includes informational pages and can generate a PDF report for a prediction.

> **Medical disclaimer:** This project is for educational and demonstration purposes only. It is not a medical device and does not provide a diagnosis or replace advice from a qualified healthcare professional. Do not use its predictions to make medical decisions.

## Project status and files needed

The repository currently does **not** contain the files that the application expects at startup:

- `heart_disease_model.pkl`
- `scaler.pkl`

As a result, `app.py` will fail while loading the model until these artifacts are provided. The training script, `prediction.py`, also expects `heart_cleveland_upload.csv` in the project root; that dataset is not currently included either. Obtain the dataset from a source you are authorized to use, and verify its schema before training. Do not commit sensitive or restricted data, or model artifacts from an untrusted source.

The repository includes `requirements.txt` and a `Procfile` for Gunicorn-based deployment. The README documents the intended setup, but the missing dataset and model artifacts must be addressed before the application can run.

## Requirements

- Python 3.9 or later (use a Python version supported by the installed scikit-learn release)
- pip
- The dataset and generated model files described above

## Setup

1. Clone the repository and change into its directory:

   ```bash
   git clone https://github.com/tarun003k/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   On macOS/Linux:

   ```bash
   source .venv/bin/activate
   ```

   On Windows PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Train the model

Place a compatible `heart_cleveland_upload.csv` file in the project root. The current training script expects a `condition` target column and the 13 feature columns used by the prediction form. Then run:

```bash
python prediction.py
```

The script trains a Random Forest classifier and writes `heart_disease_model.pkl` and `scaler.pkl` in the current directory. Keep both generated files alongside `app.py`; the application loads them during startup. Only load pickle files you trust, since unpickling an untrusted file can execute code.

The model's evaluation output is printed by the training script. Treat its reported metrics as a basic holdout estimate, not evidence of clinical validity.

## Run the web app

Once both model files exist in the project root:

```bash
python app.py
```

Open <http://127.0.0.1:5000> in your browser. The repository also contains a `Procfile` configured to run the app with Gunicorn:

```bash
gunicorn app:app
```

## Application pages

- `/` — home page
- `/prediction` — prediction form
- `/about` — project information
- `/trends` — trends page
- `/article` — article page
- `/team` — team page
- `/predict` — handles submitted prediction form data
- `/download_report` — downloads a PDF report for the current session's prediction

## Prediction inputs

The form collects 13 fields: age, sex, chest-pain type (`cp`), resting blood pressure (`trestbps`), cholesterol (`chol`), fasting blood sugar (`fbs`), resting ECG (`restecg`), maximum heart rate (`thalach`), exercise-induced angina (`exang`), ST depression (`oldpeak`), ST-segment slope (`slope`), number of major vessels (`ca`), and thalassemia result (`thal`). Input encoding and feature order must match the dataset/model used for training.

## Important deployment notes

- Replace the placeholder Flask session secret in `app.py` with a strong secret supplied through an environment variable before deploying. Do not commit production secrets.
- Disable Flask debug mode in production. Use a production WSGI server such as the Gunicorn command in the `Procfile`.
- Validate input ranges and model/data compatibility before relying on predictions; the current server primarily checks that values are present and numeric.
- Risk estimates are not a substitute for clinical assessment. Have a qualified professional evaluate any health concerns.

## License

No license file is currently included. Contact the repository owner before reusing or distributing this project.
