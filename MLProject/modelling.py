"""
modelling.py
============
Kriteria 2 - Basic
Melatih model Random Forest menggunakan MLflow autolog.
Artefak disimpan secara lokal di MLflow Tracking UI.

Penggunaan:
    mlflow ui                        # jalankan di terminal lain
    python modelling.py
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
import mlflow.sklearn


# ── Konfigurasi ────────────────────────────────────────────────
TRAIN_PATH   = "../preprocessing/titanic_preprocessing/titanic_train.csv"
TEST_PATH    = "../preprocessing/titanic_preprocessing/titanic_test.csv"
EXPERIMENT   = "Titanic_Classification"
RUN_NAME     = "RandomForest_Autolog"


def load_data():
    """Memuat dataset train dan test hasil preprocessing."""
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    X_train = train.drop(columns=["Survived"])
    y_train = train["Survived"]
    X_test  = test.drop(columns=["Survived"])
    y_test  = test["Survived"]

    print(f"Data training : {X_train.shape}")
    print(f"Data testing  : {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train():
    """Pipeline pelatihan model dengan MLflow autolog."""

    # Set tracking URI ke lokal
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(EXPERIMENT)

    # Aktifkan autolog
    mlflow.sklearn.autolog()

    print("=" * 50)
    print(" TRAINING MODEL - BASIC (Autolog)")
    print("=" * 50)

    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run(run_name=RUN_NAME):

        # Definisi model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

        # Training
        print("\nMelatih model Random Forest ...")
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)

        # Evaluasi
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)

        print(f"\n✅ Hasil Evaluasi:")
        print(f"   Accuracy  : {acc:.4f}")
        print(f"   F1-Score  : {f1:.4f}")
        print(f"   Precision : {prec:.4f}")
        print(f"   Recall    : {rec:.4f}")

    print("\n✅ Run selesai! Cek MLflow UI di http://127.0.0.1:5000")


if __name__ == "__main__":
    train()
