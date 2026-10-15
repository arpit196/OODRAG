"""Monotonic wall-clock timings, including failed and skipped operations."""
from contextlib import contextmanager
from time import perf_counter


@contextmanager
def measure(metrics: dict, name: str):
    started = perf_counter()
    entry = {"duration_ms": None, "status": "running"}
    metrics[name] = entry
    try:
        yield
    except Exception:
        entry["status"] = "error"
        raise
    else:
        entry["status"] = "completed"
    finally:
        entry["duration_ms"] = round((perf_counter() - started) * 1000, 3)


def skipped_metrics(names):
    return {name: {"duration_ms": None, "status": "skipped"} for name in names}
