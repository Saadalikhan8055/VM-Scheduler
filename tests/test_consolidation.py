from src.entities import PM, PMPool, VM
from src.consolidation import consolidate


def test_consolidate_aborts_when_any_vm_cannot_be_moved():
    source_pm = PM(pm_id="source")
    target_pm = PM(pm_id="target")
    pool = PMPool([source_pm, target_pm])

    vm_a = VM(vm_id=1, cpu_request=0.1, memory_request=0.1, arrival_time=0.0, vm_type=0)
    vm_b = VM(vm_id=2, cpu_request=0.1, memory_request=0.1, arrival_time=1.0, vm_type=0)

    source_pm.add_vm(vm_a)
    source_pm.add_vm(vm_b)
    target_pm.cpu_used = 0.74
    target_pm.memory_used = 0.74

    migrated_count, idle_ids = consolidate(pool, upper_threshold=0.75)

    assert migrated_count == 0
    assert idle_ids == []
    assert [vm.vm_id for vm in source_pm.hosted_vms] == [1, 2]
    assert target_pm.hosted_vms == []


def test_consolidate_respects_cpu_intensive_cap():
    source_pm = PM(pm_id="source")
    target_pm = PM(pm_id="target")
    pool = PMPool([source_pm, target_pm])

    vm = VM(vm_id=3, cpu_request=0.1, memory_request=0.1, arrival_time=0.0, vm_type=2)
    source_pm.add_vm(vm)
    target_pm.cpu_used = 0.7
    target_pm.memory_used = 0.1

    migrated_count, idle_ids = consolidate(pool, upper_threshold=0.75, cpu_intensive_cap=0.25)

    assert migrated_count == 0
    assert idle_ids == []
    assert source_pm.hosted_vms == [vm]
    assert target_pm.hosted_vms == []
