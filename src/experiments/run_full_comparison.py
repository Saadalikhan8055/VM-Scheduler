from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

os.chdir(PROJECT_ROOT)

from baselines import best_fit, first_fit, round_robin
from consolidation import consolidate, simulate_vm_churn
from evaluation import SPECPOWER_TABLE, energy_consumption
from preliminary_scheduling import (
    build_job_queue,
    classify_vm_types,
    load_data,
    simulate_placement,
)


def _summarize_pool(pool, algorithm_name: str) -> dict:
    """Collect the common comparison metrics for a pool."""
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
        "Algorithm": algorithm_name,
        "Active PMs": len(active),
        "Idle PMs": len(idle),
        "Total Energy (watts)": round(energy, 2),
        "Avg CPU Utilization": round(active_cpu, 4),
        "Avg Memory Utilization": round(active_memory, 4),
    }


def run_experiment1(vms) -> pd.DataFrame:
    """Compare placement strategies on the identical full VM set (fair initial placement comparison)."""
    rows = []
    rows.append(_summarize_pool(simulate_placement(vms, 700)[0], "our_algorithm_initial_placement"))
    rows.append(_summarize_pool(round_robin(vms, 700), "round_robin"))
    rows.append(_summarize_pool(first_fit(vms, 700), "first_fit"))
    rows.append(_summarize_pool(best_fit(vms, 700), "best_fit"))
    return pd.DataFrame(rows)


def run_experiment2(vms) -> pd.DataFrame:
    """Isolate the benefit of our algorithm's consolidation stage after simulated churn."""
    pool_before, _ = simulate_placement(vms, 700)
    simulate_vm_churn(pool_before)

    before_metrics = {
        "Algorithm": "our_algorithm_before_consolidation",
        "Active PMs": len(pool_before.active_pms()),
        "Idle PMs": len(pool_before.idle_pms()),
        "Total Energy (watts)": round(energy_consumption(pool_before, SPECPOWER_TABLE), 2),
        "Avg CPU Utilization": round(
            sum(pm.current_utilization()[0] for pm in pool_before.active_pms())
            / len(pool_before.active_pms())
            if pool_before.active_pms()
            else 0.0,
            4,
        ),
        "Avg Memory Utilization": round(
            sum(pm.current_utilization()[1] for pm in pool_before.active_pms())
            / len(pool_before.active_pms())
            if pool_before.active_pms()
            else 0.0,
            4,
        ),
    }

    pool_after = pool_before
    consolidate(pool_after)

    after_metrics = {
        "Algorithm": "our_algorithm_after_consolidation",
        "Active PMs": len(pool_after.active_pms()),
        "Idle PMs": len(pool_after.idle_pms()),
        "Total Energy (watts)": round(energy_consumption(pool_after, SPECPOWER_TABLE), 2),
        "Avg CPU Utilization": round(
            sum(pm.current_utilization()[0] for pm in pool_after.active_pms())
            / len(pool_after.active_pms())
            if pool_after.active_pms()
            else 0.0,
            4,
        ),
        "Avg Memory Utilization": round(
            sum(pm.current_utilization()[1] for pm in pool_after.active_pms())
            / len(pool_after.active_pms())
            if pool_after.active_pms()
            else 0.0,
            4,
        ),
    }

    return pd.DataFrame([before_metrics, after_metrics])


def save_plots(df: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    """Save bar charts for energy and active PM comparisons."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.bar(df["Algorithm"], df["Total Energy (watts)"], color="steelblue")
    plt.ylabel("Total Energy (watts)")
    plt.title(f"{prefix} Energy Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_energy_chart.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(df["Algorithm"], df["Active PMs"], color="forestgreen")
    plt.ylabel("Active PMs")
    plt.title(f"{prefix} Active PMs Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_active_pms_chart.png", dpi=150)
    plt.close()


def main() -> None:
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    _, raw_df, svm_model, encoder_model = load_data()
    vms = build_job_queue(raw_df)
    classify_vm_types(vms, raw_df, encoder_model, svm_model)

    print(
        "Experiment 1 compares placement strategies on an identical workload "
        "(fair comparison of initial placement quality)."
    )
    experiment1_df = run_experiment1(vms)
    experiment1_df.to_csv(output_dir / "experiment1_initial_placement_comparison.csv", index=False)
    save_plots(experiment1_df, output_dir, "experiment1")
    print(experiment1_df.to_string(index=False))
    print(
        f"\nSaved Experiment 1 CSV to {output_dir / 'experiment1_initial_placement_comparison.csv'}"
    )
    print(f"Saved Experiment 1 energy chart to {output_dir / 'experiment1_energy_chart.png'}")
    print(f"Saved Experiment 1 active PMs chart to {output_dir / 'experiment1_active_pms_chart.png'}")

    print(
        "\nExperiment 2 isolates the benefit of our algorithm's consolidation/migration stage "
        "on top of its own initial placement, after simulating VM churn."
    )
    experiment2_df = run_experiment2(vms)
    experiment2_df.to_csv(output_dir / "experiment2_consolidation_benefit.csv", index=False)
    save_plots(experiment2_df, output_dir, "experiment2")
    print(experiment2_df.to_string(index=False))
    print(
        f"\nSaved Experiment 2 CSV to {output_dir / 'experiment2_consolidation_benefit.csv'}"
    )
    print(f"Saved Experiment 2 chart to {output_dir / 'experiment2_consolidation_benefit_chart.png'}")


if __name__ == "__main__":
    main()
