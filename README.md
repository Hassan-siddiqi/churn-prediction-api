# Churn Prediction API (FastAPI + Logistic Regression)

A beginner-friendly Machine Learning project that predicts customer churn using a trained Logistic Regression model and serves predictions via FastAPI.

## Dataset
Telco Customer Churn (public Kaggle dataset).  
We use only numeric/simple columns to keep it easy:
- tenure
- MonthlyCharges
- TotalCharges
- SeniorCitizen
Target:
- Churn (Yes/No)

## Project Structure
- `src/train.py` -> trains the model and saves files in `models/`
- `src/app.py` -> FastAPI app that loads the saved model and predicts
- `data/churn.csv` -> dataset CSV

## Setup & Run
```bash
pip install -r requirments.txt
python src/train.py
python -m uvicorn src.app:app --reload
