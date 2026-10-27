from __future__ import annotations

from typing import List

from entities import PM, PMPool, VM


def round_robin(vms: list[VM], num_pms: int, upper_threshold: float = 0.75) -> PMPool:
    """Assign VMs in a rotating order across PMs, skipping PMs that cannot host a VM."""
    pool = PMPool([PM(pm_id=f"pm_{i+1}") for i in range(num_pms)])
    pms = pool.pms

    for vm in vms:
        placed = False
        for offset in range(len(pms)):
            pm = pms[(offset + 0) % len(pms)]
            if pm.can_host(vm, upper_threshold=upper_threshold):
                pm.add_vm(vm)
                placed = True
                break

        if not placed:
            # Leave the VM unplaced if no PM can host it under the threshold.
            continue

    return pool


def first_fit(vms: list[VM], num_pms: int, upper_threshold: float = 0.75) -> PMPool:
    """Assign each VM to the first PM in creation order that can host it."""
    pool = PMPool([PM(pm_id=f"pm_{i+1}") for i in range(num_pms)])

    for vm in vms:
        placed = False
        for pm in pool.pms:
            if pm.can_host(vm, upper_threshold=upper_threshold):
                pm.add_vm(vm)
                placed = True
                break

        if not placed:
            continue

    return pool


def best_fit(vms: list[VM], num_pms: int, upper_threshold: float = 0.75) -> PMPool:
    """Assign each VM to the PM whose utilization would be highest after placement."""
    pool = PMPool([PM(pm_id=f"pm_{i+1}") for i in range(num_pms)])

    for vm in vms:
        best_pm = None
        best_post_util = None

        for pm in pool.pms:
            if not pm.can_host(vm, upper_threshold=upper_threshold):
                continue

            cpu_ratio, memory_ratio = pm.current_utilization()
            projected_cpu = cpu_ratio + (vm.cpu_request / pm.cpu_capacity)
            projected_memory = memory_ratio + (vm.memory_request / pm.memory_capacity)
            post_util = max(projected_cpu, projected_memory)

            if best_pm is None or post_util > best_post_util:
                best_pm = pm
                best_post_util = post_util

        if best_pm is not None:
            best_pm.add_vm(vm)

    return pool
