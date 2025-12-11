# metrics_saver.py
from __future__ import annotations

import time
import statistics
import atexit
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver


# ------------------------------
# Metrics storage
# ------------------------------

@dataclass
class OpStats:
    count: int
    p50: float
    p95: float
    p99: float
    min: float
    max: float
    avg: float


class MetricsRecorder:
    def __init__(self):
        self.samples: Dict[str, List[float]] = defaultdict(list)

    def record(self, op: str, ms: float):
        self.samples[op].append(ms)

    def summary(self) -> Dict[str, OpStats]:
        out = {}
        for op, vals in self.samples.items():
            if not vals:
                continue
            vals_sorted = sorted(vals)

            def pct(x: float):
                idx = min(len(vals_sorted) - 1, int(len(vals_sorted) * x) - 1)
                return vals_sorted[idx]

            out[op] = OpStats(
                count=len(vals_sorted),
                p50=pct(0.50),
                p95=pct(0.95),
                p99=pct(0.99),
                min=vals_sorted[0],
                max=vals_sorted[-1],
                avg=statistics.fmean(vals_sorted),
            )
        return out

    def print_summary(self):
        print("\n=== Checkpointer Metrics Summary ===")
        for op, s in self.summary().items():
            print(
                f"{op:<18} count={s.count:<5} "
                f"p50={s.p50:.3f}ms p95={s.p95:.3f}ms p99={s.p99:.3f}ms "
                f"min={s.min:.3f}ms max={s.max:.3f}ms avg={s.avg:.3f}ms"
            )


GLOBAL_METRICS = MetricsRecorder()


# ------------------------------
# Wrapper Saver
# ------------------------------

class InstrumentedSaver(BaseCheckpointSaver):
    """
    Wraps your AerospikeSaver *without modifying it*.
    """

    def __init__(self, inner: BaseCheckpointSaver, prefix="cp"):
        self.inner = inner
        self.prefix = prefix

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def _time(self, name: str, fn, *args, **kwargs):
        start = time.perf_counter_ns()
        result = fn(*args, **kwargs)
        dur = (time.perf_counter_ns() - start) / 1e6
        GLOBAL_METRICS.record(f"{self.prefix}.{name}", dur)
        return result

    # ---- override the saver operations ----

    def put(self, config, checkpoint, metadata):
        return self._time("put", self.inner.put, config, checkpoint, metadata)

    def get_tuple(self, config):
        return self._time("get_tuple", self.inner.get_tuple, config)

    def put_writes(self, config, writes):
        return self._time("put_writes", self.inner.put_writes, config, writes)

    def get_writes(self, config):
        return self._time("get_writes", self.inner.get_writes, config)

    def list(self, config, before=None, limit=None):
        return self._time("list", self.inner.list, config, before=before, limit=limit)


# print metrics on exit automatically
def _on_exit():
    GLOBAL_METRICS.print_summary()

atexit.register(_on_exit)
