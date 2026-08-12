"""
Heart Disease Logistic Regression - SageMaker script-mode entry point.

The model is the same from-scratch NumPy implementation used in
`heart_disease_lr_analysis.ipynb` (sigmoid + binary cross-entropy + L2 +
batch gradient descent). scikit-learn is NOT used for training; the SKLearn
container is used only as a Python runtime.

The script:
  1. reads heart_train.csv / heart_test.csv from the SageMaker input channels
     (features are already standardized by the local notebook),
  2. trains regularized logistic regression with gradient descent,
  3. evaluates accuracy / precision / recall / F1 on train and test,
  4. writes model.npz and metrics.json to SM_MODEL_DIR.

No endpoint is created and no deployment service is used.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Model (identical to the local notebook)
# --------------------------------------------------------------------------
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def predict_proba(w, b, X):
    return sigmoid(X @ w + b)


def predict(w, b, X, threshold=0.5):
    return (predict_proba(w, b, X) >= threshold).astype(float)


def compute_cost(w, b, X, y, lam=0.0):
    m = X.shape[0]
    f = np.clip(predict_proba(w, b, X), 1e-8, 1 - 1e-8)
    ce = -(1 / m) * np.sum(y * np.log(f) + (1 - y) * np.log(1 - f))
    return ce + (lam / (2 * m)) * np.sum(w ** 2)


def compute_gradient(w, b, X, y, lam=0.0):
    m = X.shape[0]
    error = predict_proba(w, b, X) - y
    dj_dw = (1 / m) * (X.T @ error) + (lam / m) * w
    dj_db = (1 / m) * np.sum(error)
    return dj_dw, dj_db


def gradient_descent(X, y, w_init, b_init, alpha, lam, num_iters, print_every=2000):
    w = w_init.copy()
    b = b_init
    J_history = []
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(w, b, X, y, lam)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        J_history.append(compute_cost(w, b, X, y, lam))
        if print_every > 0 and (i % print_every == 0 or i == num_iters - 1):
            print(f"Iteration {i:6d}: J_reg(w, b) = {J_history[-1]:.6f}", flush=True)
    return w, b, J_history


def evaluate(y_true, y_pred):
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    tn = float(np.sum((y_pred == 0) & (y_true == 0)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"accuracy": round(accuracy, 4), "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4),
            "TP": tp, "TN": tn, "FP": fp, "FN": fn}


# --------------------------------------------------------------------------
def load_split(channel_dir, filename):
    df = pd.read_csv(os.path.join(channel_dir, filename))
    y = df["target"].values.astype(float)
    X = df.drop(columns=["target"]).values.astype(float)
    return X, y, [c for c in df.columns if c != "target"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--num-iters", type=int, default=20000)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model-dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "./model"))
    parser.add_argument("--train", type=str,
                        default=os.environ.get("SM_CHANNEL_TRAIN", "./data"))
    parser.add_argument("--test", type=str,
                        default=os.environ.get("SM_CHANNEL_TEST", "./data"))
    args = parser.parse_args()

    print("=" * 70)
    print("Heart Disease - Logistic Regression from scratch (NumPy) on SageMaker")
    print(f"numpy {np.__version__} | pandas {pd.__version__}")
    print(f"alpha={args.alpha} num_iters={args.num_iters} lambda={args.lam}")
    print("=" * 70, flush=True)

    X_train, y_train, features = load_split(args.train, "heart_train.csv")
    X_test, y_test, _ = load_split(args.test, "heart_test.csv")
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Features: {features}", flush=True)

    w, b, J_history = gradient_descent(
        X_train, y_train,
        np.zeros(X_train.shape[1]), 0.0,
        alpha=args.alpha, lam=args.lam, num_iters=args.num_iters,
    )

    metrics = {
        "train": evaluate(y_train, predict(w, b, X_train, args.threshold)),
        "test": evaluate(y_test, predict(w, b, X_test, args.threshold)),
        "final_cost": round(float(J_history[-1]), 6),
        "weight_norm": round(float(np.linalg.norm(w)), 6),
        "hyperparameters": {"alpha": args.alpha, "num_iters": args.num_iters,
                            "lambda": args.lam, "threshold": args.threshold},
        "features": features,
        "coefficients": {f: round(float(v), 6) for f, v in zip(features, w)},
        "bias": round(float(b), 6),
    }

    print("\nTRAIN metrics:", json.dumps(metrics["train"]))
    print("TEST  metrics:", json.dumps(metrics["test"]))
    print(f"||w|| = {metrics['weight_norm']} | b = {metrics['bias']}", flush=True)

    os.makedirs(args.model_dir, exist_ok=True)
    np.savez(os.path.join(args.model_dir, "model.npz"),
             w=w, b=b, features=np.array(features),
             J_history=np.array(J_history))
    with open(os.path.join(args.model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nArtifacts written to {args.model_dir}: model.npz, metrics.json")
    print("TRAINING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
