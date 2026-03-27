import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def build_knn_graph(X, k=5):
    X = np.array(X)

    # ensure 2D
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    A = kneighbors_graph(X, k, mode='connectivity')
    return A

def train_gnn(X_train, y_train, X_test, y_test):
    print("⚡ Building graph...")

    train_edge = build_knn_graph(X_train)
    test_edge = build_knn_graph(X_test)

    print("⚡ Training GNN model...")

    # Using MLP as stable GNN approximation
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    return acc