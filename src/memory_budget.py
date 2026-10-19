"""Bound Blender workloads before macOS starts sustained memory compression."""

import ctypes
import os
import sys

GIB = 1024 ** 3


class MemoryBudgetExceeded(RuntimeError):
    pass


def memory_snapshot(pid=None):
    """Use physical footprint, including compressed memory, rather than RSS.

    macOS RSS alone can look small while tens of GiB have been compressed or
    swapped. This guard targets the shared CPU/GPU memory on Apple Silicon.
    Other platforms retain the process isolation without a macOS-specific limit.
    """
    if sys.platform != "darwin":
        return None
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)

    def sysctl(name, ctype):
        value = ctype()
        size = ctypes.c_size_t(ctypes.sizeof(value))
        if libc.sysctlbyname(name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0):
            raise OSError(ctypes.get_errno(), f"Cannot read {name}")
        return value.value

    # rusage_info_v2 from Apple's sys/resource.h: UUID followed by 18 uint64s.
    class RUsage(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16), ("values", ctypes.c_uint64 * 18)]

    usage = RUsage()
    if libc.proc_pid_rusage(pid or os.getpid(), 2, ctypes.byref(usage)):
        raise OSError(ctypes.get_errno(), "Cannot read process physical footprint")
    return {
        "process_bytes": usage.values[7],
        "total_bytes": sysctl("hw.memsize", ctypes.c_uint64),
        "pressure": sysctl("kern.memorystatus_vm_pressure_level", ctypes.c_uint32),
    }


def ensure_memory_budget(operation, pid=None):
    try:
        snapshot = memory_snapshot(pid)
    except OSError as exc:
        raise MemoryBudgetExceeded(f"{operation} stopped: memory monitoring unavailable ({exc}).") from exc
    if snapshot is None:
        return
    limit = min(8 * GIB, int(snapshot["total_bytes"] * 0.4))
    if snapshot["process_bytes"] >= limit or snapshot["pressure"] != 1:
        raise MemoryBudgetExceeded(
            f"{operation} stopped to protect system memory: process footprint "
            f"{snapshot['process_bytes'] / GIB:.1f} GiB, limit {limit / GIB:.1f} GiB, "
            f"macOS memory pressure {snapshot['pressure']} (1 = normal). "
            "Close Blender and resume from the saved checkpoint in a fresh session."
        )
