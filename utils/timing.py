"""
utils/timing.py
===============
Timing helpers and timed-call wrapper shared across switch pipelines.
"""

import time


def _now() -> float:
    return time.perf_counter()


def _secs(t0: float, t1: float) -> float:
    return float(max(0.0, t1 - t0))


def _timed_call(fn, *args, **kwargs):
    t0 = _now()
    out = fn(*args, **kwargs)
    t1 = _now()
    return out, _secs(t0, t1)
