"""
Lightweight memory instrumentation for diagnosing whether slow steps in event
loading/cleanup are memory-bound (RSS approaching system limits, swap in use,
major page faults climbing) as opposed to plain CPU processing or disk IO.

Call log_mem(label) at the same checkpoints as the existing timing logger.debug
calls so the two logs can be correlated line-for-line. log_mem is a no-op
(skips psutil/resource entirely) unless this logger is set to DEBUG.
"""
import logging
import os
import resource

import psutil

logger = logging.getLogger(__name__)

_process = psutil.Process(os.getpid())
_last_majflt = None


def log_mem(label: str):
    if not logger.isEnabledFor(logging.DEBUG):
        return

    global _last_majflt

    rss_gb = _process.memory_info().rss / 1e9
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()

    majflt = resource.getrusage(resource.RUSAGE_SELF).ru_majflt
    majflt_delta = majflt - _last_majflt if _last_majflt is not None else 0
    _last_majflt = majflt

    logger.debug(
        "[mem] %s: rss=%.2fGB sys_used=%.0f%% avail=%.2fGB/%.2fGB swap=%.2f/%.2fGB major_faults+=%d",
        label, rss_gb, vm.percent, vm.available / 1e9, vm.total / 1e9,
        swap.used / 1e9, swap.total / 1e9, majflt_delta,
    )
