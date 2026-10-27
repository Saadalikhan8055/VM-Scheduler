import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def extract_features(raw_csv_path: str, output_csv_path: str) -> pd.DataFrame:
    """Load raw datacenter trace data and engineer pseudo-VM features.

    The dataset does not include true per-VM IDs. Each row already represents a
    distinct 5-minute snapshot of aggregate resource usage, so we treat each row
    as an independent "pseudo-VM" sample. This lets us adapt the original
    per-VM pipeline semantics to a datacenter-level trace by using time-window
    snapshots as the unit of analysis.
    """
    df = pd.read_csv(raw_csv_path)

    # Create a synthetic vm_id column from the row index to preserve sample identity.
    df = df.reset_index(drop=True)
    df["vm_id"] = df.index

    # Rolling features are computed over the previous 5 rows only, excluding the
    # current row. This simulates short-term workload variability preceding each
    # pseudo-VM snapshot.
    rolling_window = 5
    df["cpu_rolling_max"] = df["avg_cpu"].shift(1).rolling(rolling_window, min_periods=rolling_window).max()
    df["cpu_rolling_min"] = df["avg_cpu"].shift(1).rolling(rolling_window, min_periods=rolling_window).min()
    df["cpu_rolling_std"] = df["avg_cpu"].shift(1).rolling(rolling_window, min_periods=rolling_window).std()
    df["mem_rolling_max"] = df["avg_mem"].shift(1).rolling(rolling_window, min_periods=rolling_window).max()
    df["mem_rolling_min"] = df["avg_mem"].shift(1).rolling(rolling_window, min_periods=rolling_window).min()
    df["mem_rolling_std"] = df["avg_mem"].shift(1).rolling(rolling_window, min_periods=rolling_window).std()

    # Drop the first rows where the rolling window is incomplete.
    df = df.iloc[rolling_window:].copy()

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

    df = df[["vm_id"] + feature_columns]

    # Remove any rows with missing values after rolling feature computation.
    df = df.dropna().reset_index(drop=True)

    # Normalize the ten features using MinMaxScaler.
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[feature_columns] = scaler.fit_transform(df[feature_columns])

    output_dir = os.path.dirname(output_csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save the raw unscaled features for reference and the scaled features as the
    # canonical processed dataset used by downstream clustering and autoencoding.
    raw_output_csv_path = os.path.join(output_dir, "features_raw.csv")
    df.to_csv(raw_output_csv_path, index=False)
    df_scaled.to_csv(output_csv_path, index=False)

    print_summary(df_scaled, feature_columns)
    print(f"Raw unscaled features saved to: {raw_output_csv_path}")
    print(f"Scaled features saved to: {output_csv_path}")
    return df_scaled


def print_summary(df_scaled: pd.DataFrame, feature_columns) -> None:
    print("Processed feature summary")
    print("-------------------------")
    print(f"Number of pseudo-VM samples: {df_scaled.shape[0]}")
    print("Feature ranges after MinMax scaling:")
    for col in feature_columns:
        min_val = df_scaled[col].min()
        max_val = df_scaled[col].max()
        print(f" - {col}: min={min_val:.6f}, max={max_val:.6f}")
    missing = df_scaled.isna().sum().sum()
    print(f"Missing values remaining: {missing}")


def main() -> None:
    raw_csv_path = os.path.join("data", "raw", "instance_usage_grouped_300_seconds_month.csv")
    output_csv_path = os.path.join("data", "processed", "features.csv")
    extract_features(raw_csv_path, output_csv_path)


if __name__ == "__main__":
    main()
