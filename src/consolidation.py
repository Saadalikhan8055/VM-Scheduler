from __future__ import annotations

import random
from typing import List, Tuple

from entities import PM, PMPool, VM
from evaluation import SPECPOWER_TABLE, generate_report


def consolidate(
    pool: PMPool,
    upper_threshold: float = 0.75,
    lower_threshold: float = 0.35,
    cpu_intensive_type: int = 2,
    cpu_intensive_cap: float = 0.25,
) -> tuple[int, list[str]]:
    """Consolidate VMs from underloaded PMs onto other active PMs.

    The migration is all-or-nothing per source PM: if any hosted VM cannot be
    moved to a valid target PM, the whole PM keeps its VMs untouched. This avoids
    partially consolidating a PM while leaving it in a mixed state.
    """
    active_pms = pool.active_pms()
    overloaded = [pm for pm in active_pms if any(u > upper_threshold for u in pm.current_utilization())]
    underloaded = [
        pm
        for pm in active_pms
        if all(u < lower_threshold for u in pm.current_utilization())
    ]

    before_active_count = len(active_pms)
    before_idle_count = len(pool.idle_pms())
    before_cpu_avg = (
        sum(pm.current_utilization()[0] for pm in active_pms) / len(active_pms)
        if active_pms
        else 0.0
    )
    before_mem_avg = (
        sum(pm.current_utilization()[1] for pm in active_pms) / len(active_pms)
        if active_pms
        else 0.0
    )

    migrated_count = 0
    idle_pm_ids: list[str] = []

    for source_pm in underloaded:
        tentative_moves: list[tuple[VM, PM]] = []
        can_migrate_all = True

        # All-or-nothing rule: if any VM cannot be placed safely, the whole PM
        # stays unchanged so consolidation never leaves a partially moved PM.
        for vm in list(source_pm.hosted_vms):
            target_pm = None
            candidate_pms = [
                pm for pm in pool.active_pms() if pm.pm_id != source_pm.pm_id
            ]
            candidate_pms = sorted(candidate_pms, key=lambda pm: sum(pm.current_utilization()))

            for candidate in candidate_pms:
                if not candidate.can_host(vm, upper_threshold=upper_threshold):
                    continue

                if vm.vm_type == cpu_intensive_type:
                    # Keep CPU-intensive VMs below the configured cap for the target PM.
                    projected_ratio = (candidate.count_type(cpu_intensive_type) + 1) / (
                        len(candidate.hosted_vms) + 1
                    )
                    if projected_ratio > cpu_intensive_cap:
                        continue

                target_pm = candidate
                break

            if target_pm is None:
                can_migrate_all = False
                break

            tentative_moves.append((vm, target_pm))

        if not can_migrate_all:
            # Roll back any tentative moves from this PM attempt.
            for vm, target_pm in tentative_moves:
                target_pm.remove_vm(vm)
                source_pm.add_vm(vm)
            continue

        for vm, target_pm in tentative_moves:
            source_pm.remove_vm(vm)
            target_pm.add_vm(vm)
            migrated_count += 1

        if len(source_pm.hosted_vms) == 0:
            idle_pm_ids.append(source_pm.pm_id)

    _print_consolidation_summary(
        pool,
        migrated_count,
        idle_pm_ids,
        before_active_count,
        before_idle_count,
        before_cpu_avg,
        before_mem_avg,
    )
    return migrated_count, idle_pm_ids


def _print_consolidation_summary(
    pool: PMPool,
    migrated_count: int,
    idle_pm_ids: list[str],
    before_active_count: int,
    before_idle_count: int,
    before_cpu_avg: float,
    before_mem_avg: float,
) -> None:
    """Print a concise before/after consolidation summary."""
    active = pool.active_pms()
    idle = pool.idle_pms()
    active_cpu = sum(pm.current_utilization()[0] for pm in active) / len(active) if active else 0.0
    active_memory = sum(pm.current_utilization()[1] for pm in active) / len(active) if active else 0.0

    print("Consolidation summary:")
    print(
        f"Before consolidation: {before_active_count} active PMs, {before_idle_count} idle PMs, "
        f"avg utilization {before_cpu_avg:.4f}/{before_mem_avg:.4f}"
    )
    print(
        f"After consolidation: {len(active)} active PMs, {len(idle)} idle PMs, "
        f"avg utilization {active_cpu:.4f}/{active_memory:.4f}"
    )
    print(f"VMs migrated: {migrated_count}")
    print(f"Idle PM IDs: {idle_pm_ids}")


def simulate_vm_churn(pool: PMPool, departure_fraction: float = 0.3, seed: int = 42) -> int:
    """Simulate VMs departing randomly to create realistic underloaded PMs."""
    hosted_vms = [vm for pm in pool.active_pms() for vm in pm.hosted_vms]
    if not hosted_vms:
        print("No hosted VMs available for churn simulation.")
        return 0

    rng = random.Random(seed)
    num_to_remove = max(1, int(len(hosted_vms) * departure_fraction))
    selected_vms = rng.sample(hosted_vms, k=num_to_remove)

    for vm in selected_vms:
        if vm.current_pm is not None:
            vm.current_pm.remove_vm(vm)

    active = pool.active_pms()
    active_cpu_avg = (
        sum(pm.current_utilization()[0] for pm in active) / len(active) if active else 0.0
    )
    active_mem_avg = (
        sum(pm.current_utilization()[1] for pm in active) / len(active) if active else 0.0
    )

    print(f"Removed {len(selected_vms)} VMs during churn simulation.")
    print(
        f"Average utilization after churn: CPU={active_cpu_avg:.4f}, "
        f"memory={active_mem_avg:.4f}"
    )
    return len(selected_vms)


if __name__ == "__main__":
    from preliminary_scheduling import (
        build_job_queue,
        classify_vm_types,
        load_data,
        print_summary,
        simulate_placement,
    )

    latent_df, raw_df, svm_model, encoder_model = load_data()
    vms = build_job_queue(raw_df)
    classify_vm_types(vms, raw_df, encoder_model, svm_model)
    pool, skipped = simulate_placement(vms, 700)

    print("Stage 1: After initial placement")
    print_summary(pool, vms, skipped)
    generate_report(pool, SPECPOWER_TABLE, "Energy report")

    simulate_vm_churn(pool)

    print("\nStage 2: After simulated churn")
    print(f"Active PMs: {len(pool.active_pms())}")
    print(f"Idle PMs: {len(pool.idle_pms())}")
    active = pool.active_pms()
    active_cpu_avg = (
        sum(pm.current_utilization()[0] for pm in active) / len(active) if active else 0.0
    )
    active_mem_avg = (
        sum(pm.current_utilization()[1] for pm in active) / len(active) if active else 0.0
    )
    print(f"Average CPU utilization: {active_cpu_avg:.4f}")
    print(f"Average memory utilization: {active_mem_avg:.4f}")
    generate_report(pool, SPECPOWER_TABLE, "Energy report")

    consolidated_vms, idle_ids = consolidate(pool)

    print("\nStage 3: After consolidation")
    print(f"Active PMs: {len(pool.active_pms())}")
    print(f"Idle PMs: {len(pool.idle_pms())}")
    print(f"Total VMs migrated: {consolidated_vms}")
    print(f"Idle PM IDs: {idle_ids}")
    generate_report(pool, SPECPOWER_TABLE, "Energy report")

    stage1_energy = 0.0
    stage3_energy = 0.0
    # Recompute energy from the current pool state in a lightweight way.
    from evaluation import energy_consumption

    stage3_energy = energy_consumption(pool, SPECPOWER_TABLE)

    # Recreate the initial placement state to compare Stage 1 vs Stage 3.
    _, raw_df, svm_model, encoder_model = load_data()
    stage1_vms = build_job_queue(raw_df)
    classify_vm_types(stage1_vms, raw_df, encoder_model, svm_model)
    stage1_pool, _ = simulate_placement(stage1_vms, 700)
    stage1_energy = energy_consumption(stage1_pool, SPECPOWER_TABLE)

    print("\nFinal energy comparison:")
    print("Stage 1 vs Stage 3")
    print(f"  Stage 1 energy: {stage1_energy:.2f} watts")
    print(f"  Stage 3 energy: {stage3_energy:.2f} watts")
    if stage1_energy > 0:
        reduction_pct = ((stage1_energy - stage3_energy) / stage1_energy) * 100.0
        print(f"  Energy reduction: {reduction_pct:.2f}%")
    else:
        print("  Energy reduction: 0.00%")
