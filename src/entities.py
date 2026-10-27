from __future__ import annotations

import random
from typing import List, Optional


class VM:
    """Represents a virtual machine request in the cloud scheduling simulator."""

    def __init__(
        self,
        vm_id: int,
        cpu_request: float,
        memory_request: float,
        arrival_time: float,
        vm_type: Optional[int] = None,
    ) -> None:
        self.vm_id = vm_id
        self.cpu_request = float(cpu_request)
        self.memory_request = float(memory_request)
        self.arrival_time = arrival_time
        self.vm_type = vm_type
        self.current_pm: Optional[PM] = None

    def __repr__(self) -> str:
        return (
            f"VM(vm_id={self.vm_id}, cpu_request={self.cpu_request:.3f}, "
            f"memory_request={self.memory_request:.3f}, vm_type={self.vm_type})"
        )


class PM:
    """Represents a physical machine that can host VMs."""

    def __init__(
        self,
        pm_id: str,
        cpu_capacity: float = 1.0,
        memory_capacity: float = 1.0,
    ) -> None:
        self.pm_id = pm_id
        self.cpu_capacity = float(cpu_capacity)
        self.memory_capacity = float(memory_capacity)
        self.cpu_used = 0.0
        self.memory_used = 0.0
        self.hosted_vms: List[VM] = []

    def current_utilization(self) -> tuple[float, float]:
        """Return current CPU and memory utilization ratios."""
        cpu_ratio = self.cpu_used / self.cpu_capacity if self.cpu_capacity else 0.0
        memory_ratio = self.memory_used / self.memory_capacity if self.memory_capacity else 0.0
        return cpu_ratio, memory_ratio

    def can_host(self, vm: VM, upper_threshold: float = 0.75) -> bool:
        """Check whether this PM can host the VM without exceeding thresholds."""
        projected_cpu = self.cpu_used + vm.cpu_request
        projected_memory = self.memory_used + vm.memory_request
        return (
            projected_cpu <= upper_threshold * self.cpu_capacity
            and projected_memory <= upper_threshold * self.memory_capacity
        )

    def add_vm(self, vm: VM) -> None:
        """Add a VM to this PM and update resource usage."""
        if vm in self.hosted_vms:
            return
        self.cpu_used += vm.cpu_request
        self.memory_used += vm.memory_request
        self.hosted_vms.append(vm)
        vm.current_pm = self

    def remove_vm(self, vm: VM) -> None:
        """Remove a VM from this PM and update resource usage."""
        if vm not in self.hosted_vms:
            return
        self.hosted_vms.remove(vm)
        self.cpu_used -= vm.cpu_request
        self.memory_used -= vm.memory_request
        vm.current_pm = None

    def count_type(self, vm_type: int) -> int:
        """Return the number of hosted VMs matching the requested type."""
        return sum(1 for vm in self.hosted_vms if vm.vm_type == vm_type)

    def cpu_intensive_ratio(self, cpu_intensive_type: int = 2) -> float:
        """Return the fraction of hosted VMs that are CPU-intensive."""
        total = len(self.hosted_vms)
        if total == 0:
            return 0.0
        cpu_intensive_count = self.count_type(cpu_intensive_type)
        return cpu_intensive_count / total

    def __repr__(self) -> str:
        cpu_ratio, memory_ratio = self.current_utilization()
        return (
            f"PM(pm_id={self.pm_id}, cpu={cpu_ratio:.2f}, mem={memory_ratio:.2f}, "
            f"vms={len(self.hosted_vms)})"
        )


class PMPool:
    """Manage a collection of PMs and provide scheduling helper methods."""

    def __init__(self, pms: Optional[List[PM]] = None) -> None:
        self.pms = pms or []

    def least_loaded_first(self) -> List[PM]:
        """Return PMs sorted by increasing combined CPU + memory utilization."""
        return sorted(
            self.pms,
            key=lambda pm: sum(pm.current_utilization()),
        )

    def get_overloaded(self, upper_threshold: float = 0.75) -> List[PM]:
        """Return PMs whose CPU or memory utilization exceeds the threshold."""
        return [
            pm
            for pm in self.pms
            if any(u > upper_threshold for u in pm.current_utilization())
        ]

    def get_underloaded(self, lower_threshold: float = 0.35) -> List[PM]:
        """Return PMs whose CPU and memory utilization are both below the threshold."""
        return [
            pm
            for pm in self.pms
            if all(u < lower_threshold for u in pm.current_utilization())
        ]

    def active_pms(self) -> List[PM]:
        """Return PMs that are currently hosting at least one VM."""
        return [pm for pm in self.pms if len(pm.hosted_vms) > 0]

    def idle_pms(self) -> List[PM]:
        """Return PMs with no hosted VMs."""
        return [pm for pm in self.pms if len(pm.hosted_vms) == 0]

    def __repr__(self) -> str:
        return f"PMPool(pms={len(self.pms)})"


if __name__ == "__main__":
    # Create a pool of physical machines.
    pool = PMPool([PM(pm_id=f"pm_{i+1}") for i in range(5)])

    # Create a set of VMs with random normalized resource requests.
    vms = []
    for i in range(10):
        vm = VM(
            vm_id=f"vm_{i+1}",
            cpu_request=random.uniform(0.05, 0.3),
            memory_request=random.uniform(0.05, 0.3),
            arrival_time=float(i),
        )
        vms.append(vm)

    # Assign VMs greedily to the least-loaded eligible PM.
    for vm in vms:
        for pm in pool.least_loaded_first():
            if pm.can_host(vm):
                pm.add_vm(vm)
                break

    print("Final PM utilizations:")
    for pm in pool.pms:
        cpu_ratio, memory_ratio = pm.current_utilization()
        print(
            f"{pm.pm_id}: cpu={cpu_ratio:.3f}, memory={memory_ratio:.3f}, "
            f"hosted_vms={len(pm.hosted_vms)}"
        )
