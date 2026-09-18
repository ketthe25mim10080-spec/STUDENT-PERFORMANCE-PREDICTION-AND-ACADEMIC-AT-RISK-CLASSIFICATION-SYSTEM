import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

DATA_PATH = "student_data.csv"
MODEL_PATH = "student_model.pkl"

def generate_synthetic_data(num_samples=500):
    """Generates synthetic dataset for student academic performance tracking."""
    np.random.seed(42)
    study_hours = np.random.uniform(1, 10, num_samples)
    attendance_rate = np.random.uniform(50, 100, num_samples)
    past_score = np.random.uniform(40, 100, num_samples)
    extracurricular = np.random.choice([0, 1], size=num_samples)
    sleep_hours = np.random.uniform(4, 9, num_samples)
    
    # Target logic: Pass (1) or Fail/At-Risk (0)
    score = (study_hours * 4) + (attendance_rate * 0.3) + (past_score * 0.4) + (extracurricular * 2) - (8 - sleep_hours) * 2
    status = [1 if s > 55 else 0 for s in score]
    
    df = pd.DataFrame({
        'StudyHours': np.round(study_hours, 1),
        'AttendanceRate': np.round(attendance_rate, 1),
        'PastScore': np.round(past_score, 1),
        'Extracurricular': extracurricular,
        'SleepHours': np.round(sleep_hours, 1),
        'Status': status
    })
    
    df.to_csv(DATA_PATH, index=False)
    print(f"[INFO] Synthetic dataset generated and saved to '{DATA_PATH}'.")

def train_model():
    """Preprocesses data, trains Random Forest model, and saves artifacts."""
    if not os.path.exists(DATA_PATH):
        print("[INFO] Dataset not found. Generating sample dataset...")
        generate_synthetic_data()

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['Status'])
    y = df['Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)

    predictions = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, predictions)

    print("\n--- Model Training Results ---")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, predictions))

    joblib.dump({'model': clf, 'scaler': scaler}, MODEL_PATH)
    print(f"[INFO] Trained model saved to '{MODEL_PATH}'.")

def predict_sample(study, attendance, past, extra, sleep):
    """Predicts outcome for a specific student input via CLI."""
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Model file not found. Please train the model first using '--train'.")
        return

    artifacts = joblib.load(MODEL_PATH)
    model = artifacts['model']
    scaler = artifacts['scaler']

    features = np.array([[study, attendance, past, extra, sleep]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    prob = model.predict_proba(scaled_features)[0][prediction]

    result = "PASS / NOT AT-RISK" if prediction == 1 else "FAIL / AT-RISK"
    print("\n--- Prediction Output ---")
    print(f"Result: {result} (Confidence: {prob * 100:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description="Student Performance & Risk Prediction CLI Tool")
    parser.add_argument('--generate-data', action='store_true', help="Generate synthetic dataset")
    parser.add_argument('--train', action='store_true', help="Train the model and save checkpoint")
    parser.add_argument('--predict', action='store_true', help="Run prediction on custom input")
    parser.add_argument('--study', type=float, default=5.0, help="Weekly study hours (1-10)")
    parser.add_argument('--attendance', type=float, default=75.0, help="Attendance percentage (50-100)")
    parser.add_argument('--past_score', type=float, default=65.0, help="Previous exam score (0-100)")
    parser.add_argument('--extra', type=int, choices=[0, 1], default=1, help="Extracurricular participation (1=Yes, 0=No)")
    parser.add_argument('--sleep', type=float, default=7.0, help="Average daily sleep hours (4-9)")

    args = parser.parse_args()

    if args.generate_data:
        generate_synthetic_data()
    elif args.train:
        train_model()
    elif args.predict:
        predict_sample(args.study, args.attendance, args.past_score, args.extra, args.sleep)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()