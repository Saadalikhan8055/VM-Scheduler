from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
os.chdir(PROJECT_ROOT)

from consolidation import consolidate, simulate_vm_churn
from evaluation import SPECPOWER_TABLE, energy_consumption
from preliminary_scheduling import (
    build_job_queue,
    classify_vm_types,
    load_data,
    simulate_placement,
)


def run_sensitivity_analysis() -> pd.DataFrame:
    """Sweep consolidation thresholds over a 36-point grid and summarize outcomes."""
    _, raw_df, svm_model, encoder_model = load_data()
    vms = build_job_queue(raw_df)
    classify_vm_types(vms, raw_df, encoder_model, svm_model)

    upper_thresholds = [0.65, 0.70, 0.75, 0.80]
    lower_thresholds = [0.25, 0.35, 0.45]
    cpu_intensive_caps = [0.15, 0.25, 0.35]

    rows = []
    total = len(upper_thresholds) * len(lower_thresholds) * len(cpu_intensive_caps)
    run_idx = 0

    for upper_threshold in upper_thresholds:
        for lower_threshold in lower_thresholds:
            for cpu_intensive_cap in cpu_intensive_caps:
                run_idx += 1
                print(
                    f"Running combination {run_idx}/{total}: "
                    f"upper={upper_threshold}, lower={lower_threshold}, cap={cpu_intensive_cap}"
                )

                pool, _ = simulate_placement(vms, 700)
                baseline_energy = energy_consumption(pool, SPECPOWER_TABLE)

                simulate_vm_churn(pool, departure_fraction=0.3, seed=42)
                pre_consolidation_energy = energy_consumption(pool, SPECPOWER_TABLE)

                consolidate(
                    pool,
                    upper_threshold=upper_threshold,
                    lower_threshold=lower_threshold,
                    cpu_intensive_cap=cpu_intensive_cap,
                )
                post_energy = energy_consumption(pool, SPECPOWER_TABLE)

                reduction_pct = 0.0
                if pre_consolidation_energy > 0:
                    reduction_pct = ((pre_consolidation_energy - post_energy) / pre_consolidation_energy) * 100.0

                rows.append(
                    {
                        "upper_threshold": upper_threshold,
                        "lower_threshold": lower_threshold,
                        "cpu_intensive_cap": cpu_intensive_cap,
                        "active_pms_after_consolidation": len(pool.active_pms()),
                        "energy_after_consolidation": round(post_energy, 2),
                        "energy_reduction_pct": round(reduction_pct, 4),
                    }
                )

    results_df = pd.DataFrame(rows)

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sensitivity_analysis_results.csv"
    results_df.to_csv(output_path, index=False)

    averaged = (
        results_df.groupby(["upper_threshold", "lower_threshold"], as_index=False)["energy_reduction_pct"]
        .mean()
    )
    pivot = averaged.pivot(index="lower_threshold", columns="upper_threshold", values="energy_reduction_pct")

    plt.figure(figsize=(8, 5))
    plt.imshow(pivot.to_numpy(), cmap="viridis", aspect="auto")
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.colorbar(label="Energy Reduction %")
    plt.xlabel("upper_threshold")
    plt.ylabel("lower_threshold")
    plt.title("Sensitivity of Energy Reduction to Thresholds")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            plt.text(j, i, f"{pivot.iloc[i, j]:.1f}", ha="center", va="center", color="white")
    plt.tight_layout()
    plt.savefig(output_dir / "sensitivity_heatmap.png", dpi=150)
    plt.close()

    best_row = results_df.loc[results_df["energy_reduction_pct"].idxmax()]
    worst_row = results_df.loc[results_df["energy_reduction_pct"].idxmin()]

    print("\nSensitivity analysis results saved.")
    print(results_df.to_string(index=False))
    print(f"\nBest combination: upper={best_row['upper_threshold']}, lower={best_row['lower_threshold']}, cap={best_row['cpu_intensive_cap']}, energy reduction={best_row['energy_reduction_pct']:.4f}%")
    print(f"Worst combination: upper={worst_row['upper_threshold']}, lower={worst_row['lower_threshold']}, cap={worst_row['cpu_intensive_cap']}, energy reduction={worst_row['energy_reduction_pct']:.4f}%")
    print(f"Saved CSV to {output_path}")
    print(f"Saved heatmap to {output_dir / 'sensitivity_heatmap.png'}")

    return results_df


if __name__ == "__main__":
    run_sensitivity_analysis()
