import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def cluster_pseudo_vms(input_csv: str, output_csv: str, profile_path: str, size_path: str) -> pd.DataFrame:
    """Load processed features, cluster pseudo-VM snapshots, and save labels and charts."""
    df = pd.read_csv(input_csv)

    # Use only the known 10 numeric feature columns from the scaled dataset.
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

    # The input file should already be MinMax-scaled, but re-scale again here to
    # ensure K-Means receives features on the same range.
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df["cluster_label"] = cluster_labels

    _print_cluster_summary(df)
    _save_cluster_plots(df, profile_path, size_path)

    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def _print_cluster_summary(df: pd.DataFrame) -> None:
    print("Cluster summary")
    print("---------------")

    counts = df["cluster_label"].value_counts().sort_index()
    for cluster_id, count in counts.items():
        print(f"Cluster {cluster_id}: {count} samples")

    cluster_means = df.groupby("cluster_label")[["avg_cpu", "avg_mem"]].mean()
    print("\nMean avg_cpu and avg_mem per cluster:")
    print(cluster_means.to_string())

    cpu_intensive = cluster_means["avg_cpu"].idxmax()
    remaining_clusters = cluster_means.drop(index=cpu_intensive)
    memory_intensive = remaining_clusters["avg_mem"].idxmax()
    remaining_cluster = remaining_clusters.drop(index=memory_intensive).index.item()

    print(f"\nCluster {cpu_intensive} -> CPU-intensive")
    print(f"Cluster {memory_intensive} -> memory-intensive")
    print(f"Cluster {remaining_cluster} -> light")


def _save_cluster_plots(df: pd.DataFrame, profile_path: str, size_path: str) -> None:
    cluster_means = df.groupby("cluster_label")[["avg_cpu", "avg_mem"]].mean()
    os.makedirs(os.path.dirname(profile_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    cluster_means.plot(kind="bar", ax=ax)
    ax.set_title("Mean avg_cpu and avg_mem per Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Mean value")
    ax.legend(title="Metric")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(profile_path)
    plt.close(fig)

    cluster_counts = df["cluster_label"].value_counts().sort_index()
    os.makedirs(os.path.dirname(size_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    cluster_counts.plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title("Number of Samples per Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Sample count")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(size_path)
    plt.close(fig)


def main() -> None:
    input_csv = os.path.join("data", "processed", "features.csv")
    output_csv = os.path.join("data", "processed", "labeled_features.csv")
    profile_path = os.path.join("results", "cluster_profile.png")
    size_path = os.path.join("results", "cluster_sizes.png")

    cluster_pseudo_vms(input_csv, output_csv, profile_path, size_path)


if __name__ == "__main__":
    main()
