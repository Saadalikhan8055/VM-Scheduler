import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from entities import PM, PMPool, VM

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, Any, Any]:
    """Load latent features, raw labeled features, SVM model, and encoder model."""
    latent_path = PROJECT_ROOT / "data" / "processed" / "latent_features.csv"
    raw_path = PROJECT_ROOT / "data" / "processed" / "labeled_features.csv"
    svm_path = PROJECT_ROOT / "models" / "svm_classifier.pkl"
    encoder_path = PROJECT_ROOT / "models" / "encoder_model.keras"

    latent_df = pd.read_csv(latent_path)
    raw_df = pd.read_csv(raw_path)
    svm_model = joblib.load(svm_path)
    encoder_model = tf.keras.models.load_model(encoder_path)

    return latent_df, raw_df, svm_model, encoder_model


def build_job_queue(raw_df: pd.DataFrame) -> list[VM]:
    """Create VM objects from raw feature rows, scaling resources down by 0.1."""
    vms = []
    for _, row in raw_df.iterrows():
        vm_id_int = int(float(row["vm_id"]))
        arrival_time = float(vm_id_int)

        # Scale down avg_cpu and avg_mem so many VMs can fit on a PM.
        cpu_request = float(row["avg_cpu"]) * 0.1
        memory_request = float(row["avg_mem"]) * 0.1

        vm = VM(
            vm_id=vm_id_int,
            cpu_request=cpu_request,
            memory_request=memory_request,
            arrival_time=arrival_time,
        )
        vms.append(vm)

    return sorted(vms, key=lambda vm: vm.arrival_time)


def classify_vm_types(
    vms: list[VM],
    raw_df: pd.DataFrame,
    encoder_model: Any,
    svm_model: Any,
) -> None:
    """Classify each VM by encoding its raw features and predicting with the SVM."""
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

    raw_df["vm_id"] = raw_df["vm_id"].astype(int)
    raw_map = raw_df.set_index("vm_id")[feature_columns]
    vm_ids = [vm.vm_id for vm in vms]
    raw_features = raw_map.loc[vm_ids].values.astype(float)

    # Use the encoder model to build latent representations for each VM.
    latent_features = encoder_model.predict(raw_features, batch_size=64)
    predictions = svm_model.predict(latent_features)

    for vm, vm_type in zip(vms, predictions):
        vm.vm_type = int(vm_type)


def simulate_placement(vms: list[VM], num_pms: int) -> tuple[PMPool, int]:
    """Assign VMs sequentially to the least-loaded PM that can host them."""
    pool = PMPool([PM(pm_id=f"pm_{i+1}") for i in range(num_pms)])
    skipped = 0

    for vm in vms:
        placed = False
        for pm in pool.least_loaded_first():
            if pm.can_host(vm, upper_threshold=0.75):
                pm.add_vm(vm)
                placed = True
                break

        if not placed:
            skipped += 1
            print(f"Warning: VM {vm.vm_id} could not be placed on any PM and was skipped.")

    return pool, skipped


def print_summary(pool: PMPool, vms: list[VM], skipped: int) -> None:
    """Print summary statistics about placement, utilization, and VM types."""
    active = pool.active_pms()
    idle = pool.idle_pms()
    placed_vms = [vm for vm in vms if vm.current_pm is not None]

    active_cpu = np.mean([pm.current_utilization()[0] for pm in active]) if active else 0.0
    active_memory = np.mean([pm.current_utilization()[1] for pm in active]) if active else 0.0

    type_counts = {0: 0, 1: 0, 2: 0}
    for vm in placed_vms:
        if vm.vm_type in type_counts:
            type_counts[vm.vm_type] += 1

    print("\nPlacement summary:")
    print(f"Active PMs: {len(active)}")
    print(f"Idle PMs: {len(idle)}")
    print(f"VMs placed: {len(placed_vms)}")
    print(f"VMs skipped: {skipped}")
    print(f"Average CPU utilization on active PMs: {active_cpu:.4f}")
    print(f"Average memory utilization on active PMs: {active_memory:.4f}")
    print("VM count by predicted vm_type:")
    for vm_type, count in type_counts.items():
        print(f"  type {vm_type}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preliminary VM scheduling simulator.")
    parser.add_argument(
        "--num_pms",
        type=int,
        default=50,
        help="Number of physical machines to create for placement.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    latent_df, raw_df, svm_model, encoder_model = load_data()

    # Build the scheduling job queue using normalized original avg_cpu and avg_mem scaled by 0.1.
    vms = build_job_queue(raw_df)

    # Predict VM types with the saved encoder and SVM classifier.
    classify_vm_types(vms, raw_df, encoder_model, svm_model)

    pool, skipped = simulate_placement(vms, args.num_pms)
    print_summary(pool, vms, skipped)


if __name__ == "__main__":
    main()
