import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.gnn_model import train_gnn

def run_training():
    print("🚀 Starting Training Pipeline...")

    # =========================
    # DATASET
    # =========================
    X, y = make_classification(
        n_samples=3000,
        n_features=20,
        n_classes=2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    # =========================
    # RANDOM FOREST
    # =========================
    print("\n🌲 Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100)

    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)

    print(f"✅ RF Accuracy: {rf_acc}")

    # =========================
    # GNN
    # =========================
    print("\n🧠 Training GNN...")

    # 🔥 Ensure proper format (VERY IMPORTANT)
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    if len(X_train.shape) == 1:
        X_train = X_train.reshape(-1, 1)

    if len(X_test.shape) == 1:
        X_test = X_test.reshape(-1, 1)

    # run GNN
    gnn_acc = train_gnn(X_train, y_train, X_test, y_test)

    print(f"✅ GNN Accuracy: {gnn_acc}")

    # =========================
    # HYBRID
    # =========================
    hybrid_acc = (rf_acc + gnn_acc) / 2
    print(f"\n🔷 Hybrid Accuracy: {hybrid_acc}")

    # =========================
    # SAVE MODEL
    # =========================
    joblib.dump(rf_model, "models/random_forest_model.pkl")
    print("\n💾 Model saved!")

    print("\n🏁 Training Completed!")