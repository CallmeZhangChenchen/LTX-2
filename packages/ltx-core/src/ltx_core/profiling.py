"""Lightweight wall-clock profiling for the LTX inference stack.

Enabled entirely through environment variables, so a normal run pays one
module-level boolean check:

    LTX_PROFILE=1 uv run python -m ltx_pipelines.distilled ...
    LTX_PROFILE=1 LTX_PROFILE_BLOCKS=1 ...   # also time every transformer block
    LTX_PROFILE=1 LTX_PROFILE_WARMUP=3 ...   # run the pipeline N extra times first,
                                            # letting torch.compile / CUDA graph
                                            # capture / kernel JIT settle before
                                            # the measured run

Timers synchronize the accelerator before and after the measured region, so
the numbers are real GPU/exec time rather than enqueue time. A summary table
is printed at interpreter exit.
"""

from __future__ import annotations

import atexit
import os
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Iterator

_TRUTHY = {"1", "true", "yes", "on"}
_ENABLED = os.environ.get("LTX_PROFILE", "").strip().lower() in _TRUTHY
_BLOCKS_ENABLED = _ENABLED and os.environ.get("LTX_PROFILE_BLOCKS", "").strip().lower() in _TRUTHY
try:
    _WARMUP_TARGET = int(os.environ.get("LTX_PROFILE_WARMUP", "0"))
except ValueError:
    _WARMUP_TARGET = 0

_PROCESS_START = time.perf_counter()


@dataclass
class _Stat:
    total_s: float = 0.0
    calls: int = 0


_lock = threading.Lock()
_stats: dict[str, _Stat] = {}
_children: dict[str, list[str]] = {}
_counters: dict[str, int] = {}
_flushed = False
_recording_enabled = False  # set True explicitly once warmup is complete
_PROCESS_START = time.perf_counter()


def enabled() -> bool:
    return _ENABLED


def blocks_enabled() -> bool:
    return _BLOCKS_ENABLED


def warmup_rounds() -> int:
    """How many extra full pipeline runs the caller should perform before starting measurement."""
    return max(_WARMUP_TARGET, 0)


def start_recording() -> None:
    """Called once (in the CLI main, after N warmup runs finish) to arm timers."""
    global _recording_enabled
    if not _ENABLED:
        return
    with _lock:
        _stats.clear()
        _children.clear()
        _counters.clear()
        _recording_enabled = True


def recording_is_on() -> bool:
    return _recording_enabled


def null():
    """A no-op context manager, for call sites that must stay allocation-light when disabled."""
    return nullcontext()


def _should_record() -> bool:
    return _ENABLED and _recording_enabled


def _sync() -> None:
    try:
        import torch
    except Exception:  # pragma: no cover - torch is always present in this stack
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
    except Exception:
        pass


def record(name: str, seconds: float) -> None:
    if not _should_record():
        return
    with _lock:
        stat = _stats.get(name)
        if stat is None:
            stat = _stats[name] = _Stat()
            parent = name.rsplit(".", 1)[0] if "." in name else ""
            _children.setdefault(parent, []).append(name)
        stat.total_s += seconds
        stat.calls += 1


def count(name: str, n: int = 1) -> None:
    if not _should_record():
        return
    with _lock:
        _counters[name] = _counters.get(name, 0) + n


@contextmanager
def timer(name: str) -> Iterator[None]:
    """Time a region; a no-op (beyond one bool check) when profiling is off."""
    if not _should_record():
        yield
        return
    _sync()
    start = time.perf_counter()
    try:
        yield
    finally:
        _sync()
        record(name, time.perf_counter() - start)


def report() -> str:
    if not _stats and not _counters:
        return ""
    wall = time.perf_counter() - _PROCESS_START
    lines = [
        "",
        "=" * 78,
        " LTX PROFILE   (set LTX_PROFILE=1 to enable, LTX_PROFILE_BLOCKS=1 for per-block)",
        "=" * 78,
        f"{'region':<34}{'calls':>8}{'total(s)':>12}{'avg(ms)':>12}",
        "-" * 78,
    ]
    with _lock:
        for name in sorted(_stats):
            stat = _stats[name]
            avg_ms = (stat.total_s / stat.calls) * 1000 if stat.calls else 0.0
            lines.append(f"{name:<34}{stat.calls:>8}{stat.total_s:>12.3f}{avg_ms:>12.2f}")
        if _counters:
            lines.append("-" * 78)
            lines.append("counters")
            for name in sorted(_counters):
                lines.append(f"{name:<34}{_counters[name]:>8}")
    lines.append("-" * 78)
    lines.append(f"{'process wall time':<34}{'':>8}{wall:>12.3f}")
    lines.append("note: nested regions overlap (e.g. ltx_model.blocks is inside ltx_model.forward)")
    lines.append("note: timers synchronize the device, so wall time here is higher than an unprofiled run")
    if _WARMUP_TARGET:
        lines.append(f"note: skipped {_WARMUP_TARGET} warmup runs for JIT/CUDA graph settle")
    lines.append("=" * 78)
    return "\n".join(lines)


def flush() -> None:
    global _flushed
    if _flushed:
        return
    text = report()
    if text:
        _flushed = True
        print(text, flush=True)


def reset() -> None:
    global _flushed, _recording_enabled
    with _lock:
        _stats.clear()
        _children.clear()
        _counters.clear()
        _flushed = False
        _recording_enabled = False


atexit.register(flush)
