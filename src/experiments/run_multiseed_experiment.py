from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def _summarize_pool(pool) -> dict:
    active = pool.active_pms()
    idle = pool.idle_pms()
    active_cpu = (
        sum(pm.current_utilization()[0] for pm in active) / len(active) if active else 0.0
    )
    active_memory = (
        sum(pm.current_utilization()[1] for pm in active) / len(active) if active else 0.0
    )
    energy = energy_consumption(pool, SPECPOWER_TABLE)
    return {
        "Active PMs": len(active),
        "Idle PMs": len(idle),
        "Energy (watts)": round(energy, 2),
        "Avg CPU Utilization": round(active_cpu, 4),
        "Avg Memory Utilization": round(active_memory, 4),
    }


def run_multiseed_experiment() -> pd.DataFrame:
    """Run the full placement + churn + consolidation pipeline across multiple seeds."""
    _, raw_df, svm_model, encoder_model = load_data()
    vms = build_job_queue(raw_df)
    classify_vm_types(vms, raw_df, encoder_model, svm_model)

    seeds = [42, 7, 123, 2024, 99]
    rows = []

    for idx, seed in enumerate(seeds, start=1):
        print(f"Running seed {seed} ({idx}/{len(seeds)})...")
        pool, _ = simulate_placement(vms, 700)
        before_energy = energy_consumption(pool, SPECPOWER_TABLE)

        simulate_vm_churn(pool, departure_fraction=0.3, seed=seed)
        after_churn_energy = energy_consumption(pool, SPECPOWER_TABLE)

        migrated_vms, _ = consolidate(pool)
        after_consolidation_stats = _summarize_pool(pool)

        reduction_pct = 0.0
        if before_energy > 0:
            reduction_pct = ((before_energy - after_consolidation_stats["Energy (watts)"]) / before_energy) * 100.0

        rows.append(
            {
                "Seed": seed,
                "Active PMs": after_consolidation_stats["Active PMs"],
                "Idle PMs": after_consolidation_stats["Idle PMs"],
                "Energy (watts)": after_consolidation_stats["Energy (watts)"],
                "VMs Migrated": migrated_vms,
                "Energy Reduction %": round(reduction_pct, 4),
            }
        )

    results_df = pd.DataFrame(rows)

    summary_row = {
        "Seed": "mean/std",
        "Active PMs": round(results_df["Active PMs"].mean(), 2),
        "Idle PMs": round(results_df["Idle PMs"].mean(), 2),
        "Energy (watts)": round(results_df["Energy (watts)"].mean(), 2),
        "VMs Migrated": round(results_df["VMs Migrated"].mean(), 2),
        "Energy Reduction %": round(results_df["Energy Reduction %"].mean(), 4),
    }

    std_row = {
        "Seed": "std",
        "Active PMs": round(results_df["Active PMs"].std(ddof=0), 2),
        "Idle PMs": round(results_df["Idle PMs"].std(ddof=0), 2),
        "Energy (watts)": round(results_df["Energy (watts)"].std(ddof=0), 2),
        "VMs Migrated": round(results_df["VMs Migrated"].std(ddof=0), 2),
        "Energy Reduction %": round(results_df["Energy Reduction %"].std(ddof=0), 4),
    }

    results_df = pd.concat([results_df, pd.DataFrame([summary_row, std_row])], ignore_index=True)

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "multiseed_variance_results.csv"
    results_df.to_csv(output_path, index=False)

    plt.figure(figsize=(6, 4))
    plt.boxplot(results_df.loc[results_df["Seed"].isin(seeds), "Energy Reduction %"], vert=True)
    plt.ylabel("Energy Reduction %")
    plt.title("Energy Reduction % Across Seeds")
    plt.tight_layout()
    plt.savefig(output_dir / "multiseed_energy_reduction_boxplot.png", dpi=150)
    plt.close()

    print("\nMultiseed variance results:")
    print(results_df.to_string(index=False))
    print(f"\nSaved CSV to {output_path}")
    print(f"Saved boxplot to {output_dir / 'multiseed_energy_reduction_boxplot.png'}")

    return results_df


if __name__ == "__main__":
    run_multiseed_experiment()
