# model_utils.py - load the ANN and predict; run "python model_utils.py" to retrain it from data/heart-disease.csv
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from config import DATA, MODEL, PRE, METRICS_FILE, SEED

# heart-disease.csv short codes -> the column names and values used by the app's form
RENAME = {"Age": "age", "Sex": "sex", "ChestPainType": "chest_pain", "RestingBP": "resting_bp",
          "Cholesterol": "cholesterol", "FastingBS": "fasting_bs", "RestingECG": "resting_ecg",
          "MaxHR": "max_hr", "ExerciseAngina": "exercise_angina", "Oldpeak": "oldpeak",
          "ST_Slope": "st_slope", "HeartDisease": "target"}
VALUES = {"sex": {"M": "Male", "F": "Female"},
          "chest_pain": {"ASY": "Asymptomatic", "ATA": "Atypical angina", "NAP": "Non-anginal pain", "TA": "Typical angina"},
          "fasting_bs": {0: "No", 1: "Yes"},
          "resting_ecg": {"Normal": "Normal", "ST": "ST-T abnormality", "LVH": "LVH"},
          "exercise_angina": {"N": "No", "Y": "Yes"},
          "st_slope": {"Up": "Upsloping", "Flat": "Flat", "Down": "Downsloping"}}
NUMERIC = ["age", "resting_bp", "cholesterol", "max_hr", "oldpeak"]
CATEGORICAL = list(VALUES)


@st.cache_resource
def load():
    from tensorflow.keras.models import load_model     # imported late so pages open quickly
    return load_model(MODEL), joblib.load(PRE)


@st.cache_data
def metrics():
    with open(METRICS_FILE, encoding="utf-8") as f:
        return json.load(f)


def predict(row):
    """row = dict of the 11 raw inputs -> probability of heart disease."""
    model, pre = load()
    X = pre.transform(pd.DataFrame([row]))              # 11 raw inputs -> 21 scaled / one-hot features
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return float(model.predict(X.astype(np.float32), verbose=0)[0, 0])


def signals(r):
    """Inputs linked with higher (risk) or lower (ok) disease rates in the data - association, not cause."""
    risk, ok = [], []
    (ok if r["st_slope"] == "Upsloping" else risk).append(f"ST slope: {r['st_slope']}")
    (risk if r["exercise_angina"] == "Yes" else ok).append(f"Exercise angina: {r['exercise_angina']}")
    if r["chest_pain"] == "Asymptomatic":
        risk.append("Asymptomatic chest pain type")
    elif r["chest_pain"] == "Atypical angina":
        ok.append("Atypical angina")
    (risk if r["max_hr"] < 130 else ok).append(f"Max heart rate {r['max_hr']}")
    (risk if r["oldpeak"] >= 1.5 else ok).append(f"Oldpeak {r['oldpeak']}")
    if r["fasting_bs"] == "Yes":
        risk.append("High fasting blood sugar")
    return risk, ok


def load_dataset():
    """Read the CSV and convert it to the app's column names and values."""
    df = pd.read_csv(DATA).rename(columns=RENAME)
    for col, mapping in VALUES.items():
        df[col] = df[col].map(mapping)
    df[["resting_bp", "cholesterol"]] = df[["resting_bp", "cholesterol"]].replace(0, np.nan).astype(float)  # 0 = not recorded
    return df


def train():
    """Split 80/20, fit the preprocessor, train the ANN (seed 42) and save model, preprocessor and test metrics."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                 roc_auc_score, confusion_matrix)

    df = load_dataset()
    X, y = df[NUMERIC + CATEGORICAL], df["target"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    # numeric: fill missing with the mean, then scale; categorical: fill with the most common value, then one-hot
    pre = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]), NUMERIC),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL)])
    A = pre.fit_transform(X_tr).astype(np.float32)        # fitted on training rows only
    B = pre.transform(X_te).astype(np.float32)

    def fit(seed):
        tf.keras.utils.set_random_seed(seed)
        # 20% of the training rows are held out to decide when to stop training
        Xa, Xv, ya, yv = train_test_split(A, y_tr, test_size=0.2, random_state=seed, stratify=y_tr)
        model = Sequential([Input(shape=(A.shape[1],)), Dense(24, activation="relu"), Dropout(0.3),
                            Dense(12, activation="relu"), Dense(1, activation="sigmoid")])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        hist = model.fit(Xa, ya, validation_data=(Xv, yv), epochs=100, batch_size=32, verbose=0,
                         callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)])
        return model, hist.history

    model, history = fit(SEED)
    p = model.predict(B, verbose=0).ravel()
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()

    # 10 seeds show how much the score moves between training runs
    accs = [accuracy_score(y_te, fit(s)[0].predict(B, verbose=0).ravel() >= 0.5) for s in range(10)]

    out = {"rows": int(len(df)), "test_rows": int(len(y_te)), "chol_missing": int(df["cholesterol"].isna().sum()),
           "accuracy": accuracy_score(y_te, pred), "recall": recall_score(y_te, pred),
           "precision": precision_score(y_te, pred), "f1": f1_score(y_te, pred), "roc_auc": roc_auc_score(y_te, p),
           "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
           "seed_mean": float(np.mean(accs)), "seed_std": float(np.std(accs)), "history": history}
    model.save(MODEL)
    joblib.dump(pre, PRE)
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print({k: round(v * 100, 1) for k, v in out.items() if k in ("accuracy", "recall", "precision", "f1", "roc_auc")})
    print(f"10 seeds: {out['seed_mean'] * 100:.1f}% +/- {out['seed_std'] * 100:.1f}%  |  saved model, preprocessor, metrics")


if __name__ == "__main__":
    train()
