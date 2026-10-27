import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC


def main() -> None:
    # Paths for data, output, and model artifacts.
    input_csv = os.path.join("data", "processed", "latent_features.csv")
    confusion_plot_path = os.path.join("results", "confusion_matrix.png")
    model_path = os.path.join("models", "svm_classifier.pkl")

    # Load the latent features CSV and select input features and target label.
    df = pd.read_csv(input_csv)
    feature_columns = [f"latent_{i}" for i in range(1, 6)]
    X = df[feature_columns].values
    y = df["cluster_label"].values

    # Split off 20% of the data as the held-out test set.
    X_remaining, X_test, y_remaining, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Split the remaining 80% into 70% train and 10% validation relative to the original data.
    # That means validation is 12.5% of the remaining data, and training is 87.5%.
    X_train, X_val, y_train, y_val = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.125,
        random_state=42,
        stratify=y_remaining,
    )

    print(f"Dataset sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Configure the SVM and the hyperparameter grid.
    svm = SVC(kernel="rbf", probability=False)
    param_grid = {
        "gamma": [0.001, 0.01, 0.1, 1, 10],
        "C": [0.1, 1, 10, 100],
    }

    # Use GridSearchCV on the training set to tune gamma and C with 5-fold CV.
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="accuracy",
        refit=True,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    best_gamma = grid_search.best_params_["gamma"]
    best_C = grid_search.best_params_["C"]
    print(f"Best parameters from GridSearchCV: gamma={best_gamma}, C={best_C}")

    # Refit is performed automatically on the full training set by GridSearchCV.
    best_model = grid_search.best_estimator_

    # Evaluate the best model on the held-out test set.
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred)

    print("\nTest set classification report:\n")
    print(report_text)
    print(f"Overall test accuracy: {accuracy:.4f}")

    # Build a clear summary table for accuracy and macro averages.
    summary = pd.DataFrame(
        [
            {
                "metric": "accuracy",
                "value": accuracy,
            },
            {
                "metric": "macro avg precision",
                "value": report_dict["macro avg"]["precision"],
            },
            {
                "metric": "macro avg recall",
                "value": report_dict["macro avg"]["recall"],
            },
            {
                "metric": "macro avg f1-score",
                "value": report_dict["macro avg"]["f1-score"],
            },
        ]
    )

    print("\nFinal evaluation summary:")
    print(summary.to_string(index=False, float_format="{:.4f}".format))

    # Plot and save the confusion matrix.
    cm = confusion_matrix(y_test, y_pred)
    os.makedirs(os.path.dirname(confusion_plot_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=sorted(set(y_test)),
        yticklabels=sorted(set(y_test)),
    )
    plt.title("SVM Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(confusion_plot_path)
    plt.close()

    # Save the trained SVM model to disk.
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"\nSaved trained SVM model to: {model_path}")
    print(f"Saved confusion matrix plot to: {confusion_plot_path}")


if __name__ == "__main__":
    main()
