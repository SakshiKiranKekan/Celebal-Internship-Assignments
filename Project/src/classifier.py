"""
classifier.py
--------------
Deep learning component of the pipeline: a multi-layer perceptron (neural
network) trained on the LSA embeddings to (a) predict a resume's job
category and (b) produce a calibrated confidence score used later as one
signal in the overall fit score.

Note on framework choice: this uses sklearn's MLPClassifier (a real
feed-forward neural net, trained with backprop/Adam) rather than
torch/tensorflow, so the whole project installs and trains in seconds
with no GPU and no large downloads. To swap in a torch/keras model,
implement the same `fit(X, y)` / `predict_proba(X)` interface and the
rest of the pipeline (agent, feedback) is unaffected.
"""
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


class CategoryClassifier:
    def __init__(self, hidden_layer_sizes=(256, 128), random_state=42):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=random_state,
        )
        self.label_encoder = LabelEncoder()
        self.classes_ = None

    def fit(self, X, y, verbose=True):
        y_enc = self.label_encoder.fit_transform(y)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_enc, test_size=0.15, random_state=42, stratify=y_enc
        )
        self.model.fit(X_train, y_train)
        self.classes_ = self.label_encoder.classes_
        if verbose:
            preds = self.model.predict(X_val)
            acc = accuracy_score(y_val, preds)
            print(f"[CategoryClassifier] validation accuracy: {acc:.3f}")
            print(classification_report(y_val, preds, zero_division=0, target_names=self.classes_))
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X):
        idx = self.model.predict(X)
        return self.label_encoder.inverse_transform(idx)

    def top_k(self, x_row: np.ndarray, k=3):
        proba = self.model.predict_proba(x_row.reshape(1, -1))[0]
        idx = np.argsort(proba)[::-1][:k]
        return [(self.classes_[i], float(proba[i])) for i in idx]

    def save(self, path):
        joblib.dump({"model": self.model, "classes": self.classes_,
                     "label_encoder": self.label_encoder}, path)

    @classmethod
    def load(cls, path):
        obj = cls()
        data = joblib.load(path)
        obj.model = data["model"]
        obj.classes_ = data["classes"]
        obj.label_encoder = data["label_encoder"]
        return obj
