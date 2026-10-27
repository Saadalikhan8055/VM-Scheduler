from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras import layers, models

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
os.chdir(PROJECT_ROOT)

def run_kmeans_selection() -> tuple[pd.DataFrame, Path]:
    print("Running K-Means hyperparameter comparison...")
    feature_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "features.csv")
    cols_to_drop = [c for c in ["vm_id", "cluster_label"] if c in feature_df.columns]
    X = feature_df.drop(columns=cols_to_drop).to_numpy(dtype=float)

    ks = [2, 3, 4, 5, 6]
    rows = []
    inertias = []
    silhouettes = []
    for k in ks:
        print(f"  Fitting K-Means with k={k}...")
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X)
        inertia = model.inertia_
        silhouette = silhouette_score(X, labels)
        inertias.append(inertia)
        silhouettes.append(silhouette)
        rows.append({"k": k, "inertia": inertia, "silhouette_score": silhouette})

    kmeans_df = pd.DataFrame(rows)
    plt.figure(figsize=(7, 4))
    plt.plot(kmeans_df["k"], kmeans_df["inertia"], marker="o", color="tab:blue")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("K-Means Elbow Curve")
    plt.tight_layout()
    elbow_path = PROJECT_ROOT / "results" / "kmeans_k_selection.png"
    plt.savefig(elbow_path, dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(kmeans_df["k"], kmeans_df["silhouette_score"], marker="o", color="tab:green")
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.title("K-Means Silhouette Curve")
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "results" / "kmeans_silhouette_selection.png", dpi=150)
    plt.close()

    best_k = kmeans_df.loc[kmeans_df["silhouette_score"].idxmax(), "k"]
    print(kmeans_df.to_string(index=False))
    print(f"Best k by silhouette score: {best_k}")
    return kmeans_df, elbow_path


def build_autoencoder(latent_dim: int) -> tf.keras.Model:
    input_dim = 10
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(16, activation="leaky_relu", kernel_initializer="he_normal"),
        layers.Dense(latent_dim, activation="leaky_relu", kernel_initializer="he_normal"),
        layers.Dense(16, activation="leaky_relu", kernel_initializer="he_normal"),
        layers.Dense(input_dim, activation="linear", kernel_initializer="he_normal"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss="mse")
    return model


def run_autoencoder_dim_comparison() -> pd.DataFrame:
    print("\nTraining autoencoder latent dimension comparison...")
    feature_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "features.csv")
    cols_to_drop = [c for c in ["vm_id", "cluster_label"] if c in feature_df.columns]
    X = feature_df.drop(columns=cols_to_drop).to_numpy(dtype=float)

    latent_dims = [3, 5, 8]
    rows = []
    for latent_dim in latent_dims:
        print(f"  Training autoencoder with latent_dim={latent_dim}...")
        model = build_autoencoder(latent_dim)
        history = model.fit(X, X, epochs=200, batch_size=64, verbose=0)
        final_loss = history.history["loss"][-1]
        rows.append({"latent_dim": latent_dim, "final_loss": float(final_loss)})

    autoencoder_df = pd.DataFrame(rows)
    print(autoencoder_df.to_string(index=False))
    return autoencoder_df


def run_svm_kernel_comparison() -> pd.DataFrame:
    print("\nTraining SVM kernel comparison...")
    latent_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "latent_features.csv")
    cols_to_drop = [c for c in ["vm_id", "cluster_label"] if c in latent_df.columns]
    X = latent_df.drop(columns=cols_to_drop).to_numpy(dtype=float)
    y = latent_df["cluster_label"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )

    kernels = ["rbf", "linear", "poly"]
    rows = []
    for kernel in kernels:
        print(f"  Tuning SVM with kernel={kernel}...")
        pipeline = make_pipeline(StandardScaler(), SVC(kernel=kernel, probability=False))
        grid = GridSearchCV(
            pipeline,
            param_grid={"svc__C": [1, 10, 100]},
            scoring="accuracy",
            cv=3,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        preds = best_model.predict(X_test)
        accuracy = float((preds == y_test).mean())
        from sklearn.metrics import f1_score
        f1 = float(f1_score(y_test, preds, average="weighted"))
        rows.append({"kernel": kernel, "accuracy": accuracy, "f1_score": f1})

    svm_df = pd.DataFrame(rows)
    print(svm_df.to_string(index=False))
    return svm_df


def save_summary(kmeans_df: pd.DataFrame, autoencoder_df: pd.DataFrame, svm_df: pd.DataFrame) -> Path:
    summary_rows = [
        {"experiment": "kmeans_k_selection", "k": int(row["k"]), "inertia": row["inertia"], "silhouette_score": row["silhouette_score"]}
        for _, row in kmeans_df.iterrows()
    ]
    summary_rows.extend(
        [
            {
                "experiment": "autoencoder_latent_dim_comparison",
                "latent_dim": int(row["latent_dim"]),
                "final_loss": row["final_loss"],
            }
            for _, row in autoencoder_df.iterrows()
        ]
    )
    summary_rows.extend(
        [
            {
                "experiment": "svm_kernel_comparison",
                "kernel": row["kernel"],
                "accuracy": row["accuracy"],
                "f1_score": row["f1_score"],
            }
            for _, row in svm_df.iterrows()
        ]
    )
    summary_df = pd.DataFrame(summary_rows)
    output_path = PROJECT_ROOT / "results" / "hyperparameter_comparison_summary.csv"
    summary_df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    kmeans_df, _ = run_kmeans_selection()
    autoencoder_df = run_autoencoder_dim_comparison()
    svm_df = run_svm_kernel_comparison()
    summary_path = save_summary(kmeans_df, autoencoder_df, svm_df)

    print(f"\nSaved summary CSV to {summary_path}")


if __name__ == "__main__":
    main()
