import os
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DATA_PATH = "data/churn.csv"

def main():
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Dataset columns:
    # tenure, MonthlyCharges, TotalCharges, SeniorCitizen, Churn
    df = df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn"]].copy()

    # TotalCharges sometimes comes as text / blanks -> convert to number
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Target (Churn: Yes/No -> 1/0)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    # Features (inputs)
    X = df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]]
    y = df["Churn"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale (helps logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Save model + scaler
    dump(model, "models/model.joblib")
    dump(scaler, "models/scaler.joblib")

    print(" Training done. Saved:")
    print("   - models/model.joblib")
    print("   - models/scaler.joblib")

if __name__ == "__main__":
    main()