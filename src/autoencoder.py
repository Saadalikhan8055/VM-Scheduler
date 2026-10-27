import os

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def build_autoencoder(input_dim: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build an autoencoder and return both the full model and encoder model."""
    inputs = tf.keras.Input(shape=(input_dim,), name="autoencoder_input")
    x = tf.keras.layers.Dense(
        8,
        kernel_initializer="he_normal",
        name="encoder_dense_1",
    )(inputs)
    x = tf.keras.layers.LeakyReLU(alpha=0.1, name="encoder_leaky_relu_1")(x)
    latent = tf.keras.layers.Dense(
        5,
        kernel_initializer="he_normal",
        name="latent_layer",
    )(x)
    latent = tf.keras.layers.LeakyReLU(alpha=0.1, name="latent_leaky_relu")(latent)
    x = tf.keras.layers.Dense(8, activation="relu", name="decoder_dense_1")(latent)
    outputs = tf.keras.layers.Dense(input_dim, activation="sigmoid", name="decoder_output")(x)

    autoencoder = tf.keras.Model(inputs=inputs, outputs=outputs, name="autoencoder")
    encoder = tf.keras.Model(inputs=inputs, outputs=latent, name="encoder")
    return autoencoder, encoder


def plot_training_history(history: tf.keras.callbacks.History, output_path: str) -> None:
    """Plot loss and accuracy curves from training history."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    epochs = range(1, len(history.history["loss"]) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history.history["loss"], label="train_loss")
    if "val_loss" in history.history:
        plt.plot(epochs, history.history["val_loss"], label="val_loss")

    if "accuracy" in history.history:
        plt.plot(epochs, history.history["accuracy"], label="train_accuracy")
    if "val_accuracy" in history.history:
        plt.plot(epochs, history.history["val_accuracy"], label="val_accuracy")

    plt.title("Autoencoder Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    input_csv = os.path.join("data", "processed", "labeled_features.csv")
    training_plot_path = os.path.join("results", "autoencoder_training.png")
    latent_output_csv = os.path.join("data", "processed", "latent_features.csv")
    encoder_model_path = os.path.join("models", "encoder_model.keras")

    df = pd.read_csv(input_csv)

    feature_columns = [
        "avg_cpu",
        "avg_mem",
        "avg_assigned_mem",
        "avg_cycles_per_instruction",
        "cpu_rolling_max",
        "cpu_rolling_min",
        "cpu_rolling_std",
        "mem_rolling_max",
        "mem_rolling_min",
        "mem_rolling_std",
    ]

    X = df[feature_columns].values

    print("Autoencoder input range before training: min=", X.min(), "max=", X.max())
    if X.min() < 0 or X.max() > 1:
        print("Input values outside [0, 1] detected; applying MinMaxScaler safeguard.")
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        print("Rescaled input range: min=", X.min(), "max=", X.max())

    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42, shuffle=True)

    autoencoder, encoder = build_autoencoder(input_dim=len(feature_columns))
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss="mse", metrics=["accuracy"])

    history = autoencoder.fit(
        X_train,
        X_train,
        epochs=500,
        batch_size=64,
        validation_data=(X_val, X_val),
        verbose=1,
    )

    print("\nFirst 5 epochs train loss:", history.history["loss"][:5])
    if "val_loss" in history.history:
        print("First 5 epochs val loss:", history.history["val_loss"][:5])

    final_train_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1] if "val_loss" in history.history else None
    print(f"Final training loss: {final_train_loss:.6f}")
    if final_val_loss is not None:
        print(f"Final validation loss: {final_val_loss:.6f}")

    plot_training_history(history, training_plot_path)

    latent_values = encoder.predict(X, batch_size=64)
    latent_columns = [f"latent_{i+1}" for i in range(latent_values.shape[1])]
    latent_std = latent_values.std(axis=0)
    print("\nLatent dimension standard deviations:")
    for dim, std_value in zip(latent_columns, latent_std):
        print(f"{dim}: {std_value:.6f}")

    latent_df = pd.DataFrame(latent_values, columns=latent_columns)
    latent_df["vm_id"] = df["vm_id"].values
    latent_df["cluster_label"] = df["cluster_label"].values
    latent_df = latent_df[["vm_id", "cluster_label"] + latent_columns]

    os.makedirs(os.path.dirname(latent_output_csv), exist_ok=True)
    latent_df.to_csv(latent_output_csv, index=False)

    os.makedirs(os.path.dirname(encoder_model_path), exist_ok=True)
    encoder.save(encoder_model_path)


if __name__ == "__main__":
    main()
