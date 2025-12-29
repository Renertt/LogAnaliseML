from sklearn.ensemble import IsolationForest
import joblib
import os

def train_isolation_forest(X, model_path, contamination, estimators, n_jobs):
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=estimators,
        n_jobs=n_jobs
        )
    model.fit(X)
    joblib.dump(model, model_path)
    return model

def predict_anomalies(model, X):
    preds = model.predict(X)
    scores = model.score_samples(X)
    return preds, scores

def load_model(model_path):
    return joblib.load(model_path)