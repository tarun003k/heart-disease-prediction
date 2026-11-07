import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASEDIR = Path(__file__).resolve().parent
DATASET_PATH = BASEDIR / 'heart_cleveland_upload.csv'
MODEL_PATH = BASEDIR / 'heart_disease_model.pkl'
SCALER_PATH = BASEDIR / 'scaler.pkl'


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f'Dataset not found: {DATASET_PATH}. Add the heart disease CSV before training.')

    heart = pd.read_csv(DATASET_PATH)
    if 'condition' in heart.columns:
        heart_df = heart.rename(columns={'condition': 'target'})
    elif 'target' in heart.columns:
        heart_df = heart.copy()
    else:
        raise ValueError('Dataset must contain a "condition" or "target" column for training.')

    X = heart_df.drop(columns='target')
    y = heart_df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print('Classification Report:\n', classification_report(y_test, y_pred))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

    with MODEL_PATH.open('wb') as model_file:
        pickle.dump(model, model_file)
    with SCALER_PATH.open('wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    print(f'Model saved to: {MODEL_PATH}')
    print(f'Scaler saved to: {SCALER_PATH}')


if __name__ == '__main__':
    main()

