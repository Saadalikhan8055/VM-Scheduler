from __future__ import annotations

from typing import Dict

from entities import PMPool


SPECPOWER_TABLE = {
    0: 10,
    10: 93.7,
    20: 97,
    30: 101,
    40: 105,
    50: 110,
    60: 116,
    70: 121,
    80: 125,
    90: 133,
    100: 135,
}


def _interpolate_watts(cpu_ratio: float, specpower_table: Dict[int, float]) -> float:
    """Estimate energy use for a CPU utilization ratio via linear interpolation."""
    utilization_pct = max(0.0, min(100.0, cpu_ratio * 100.0))

    if utilization_pct <= 0:
        return float(specpower_table[0])

    keys = sorted(specpower_table)
    if utilization_pct >= keys[-1]:
        return float(specpower_table[keys[-1]])

    for idx in range(len(keys) - 1):
        low_key = keys[idx]
        high_key = keys[idx + 1]
        if low_key <= utilization_pct <= high_key:
            low_watts = float(specpower_table[low_key])
            high_watts = float(specpower_table[high_key])
            if high_key == low_key:
                return low_watts
            ratio = (utilization_pct - low_key) / (high_key - low_key)
            return low_watts + (high_watts - low_watts) * ratio

    return float(specpower_table[keys[-1]])


def energy_consumption(pool: PMPool, specpower_table: Dict[int, float]) -> float:
    """Estimate total energy consumption across all PMs using SPECpower interpolation."""
    total_watts = 0.0
    for pm in pool.pms:
        if len(pm.hosted_vms) == 0:
            total_watts += 10.0
            continue

        cpu_ratio, _ = pm.current_utilization()
        total_watts += _interpolate_watts(cpu_ratio, specpower_table)

    return total_watts


def generate_report(pool: PMPool, specpower_table: Dict[int, float], stage_name: str) -> None:
    """Print a compact energy and utilization report for a pool state."""
    active = pool.active_pms()
    idle = pool.idle_pms()
    active_cpu = (
        sum(pm.current_utilization()[0] for pm in active) / len(active) if active else 0.0
    )
    active_memory = (
        sum(pm.current_utilization()[1] for pm in active) / len(active) if active else 0.0
    )
    total_energy = energy_consumption(pool, specpower_table)

    print(f"{stage_name}:")
    print(f"  Active PMs: {len(active)}")
    print(f"  Idle PMs: {len(idle)}")
    print(f"  Total energy (watts): {total_energy:.2f}")
    print(f"  Avg CPU utilization: {active_cpu:.4f}")
    print(f"  Avg memory utilization: {active_memory:.4f}")
