import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / 'Heart_disease_cleveland_new.csv'
MODEL_PATH = BASE_DIR / 'heart_disease_model.pkl'
SCALER_PATH = BASE_DIR / 'scaler.pkl'
FEATURES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
]


def main():
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f'Dataset not found: {DATASET_PATH}')

    dataset = pd.read_csv(DATASET_PATH)
    required_columns = FEATURES + ['target']
    missing = [column for column in required_columns if column not in dataset.columns]
    if missing:
        raise ValueError(f'Dataset is missing required columns: {", ".join(missing)}')

    dataset = dataset[required_columns].dropna()
    X = dataset[FEATURES]
    y = dataset['target'].astype(int)
    if y.nunique() != 2:
        raise ValueError('The target column must contain both class labels 0 and 1.')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    print(f'Holdout accuracy: {accuracy_score(y_test, predictions):.3f}')
    print(classification_report(y_test, predictions, zero_division=0))
    print('Confusion matrix:\n', confusion_matrix(y_test, predictions))

    with MODEL_PATH.open('wb') as file:
        pickle.dump(model, file)
    with SCALER_PATH.open('wb') as file:
        pickle.dump(scaler, file)
    print(f'Model saved: {MODEL_PATH}')
    print(f'Scaler saved: {SCALER_PATH}')


if __name__ == '__main__':
    main()
