"""System / process / temperature stats via psutil + smctemp."""
from __future__ import annotations

import os
import subprocess
import time

from tools._helpers import fmt_bytes


def tool_hardware(metric=""):
    """Report hardware usage (CPU / RAM / disk / process / GPU).

    Metric: cpu, ram, disk, process, top, gpu, temp, or all (default).
    """
    try:
        import psutil  # type: ignore
    except ImportError:
        return "Hardware monitoring unavailable. Install: pip install psutil"

    metric = (metric or "all").strip().lower()
    lines = []

    if metric in ("cpu", "all"):
        overall = psutil.cpu_percent(interval=0.3)
        per_core = psutil.cpu_percent(interval=0, percpu=True)
        cores = psutil.cpu_count(logical=True)
        phys = psutil.cpu_count(logical=False)
        lines.append(f"CPU: {overall:.1f}% overall ({phys} physical / {cores} logical cores)")
        lines.append("  per-core: " + ", ".join(f"{p:.0f}%" for p in per_core))

    if metric in ("ram", "memory", "all"):
        vm = psutil.virtual_memory()
        lines.append(
            f"RAM: {fmt_bytes(vm.used)} used / {fmt_bytes(vm.total)} total "
            f"({vm.percent:.1f}%), {fmt_bytes(vm.available)} available"
        )

    if metric in ("disk", "all"):
        du = psutil.disk_usage("/")
        lines.append(
            f"Disk /: {fmt_bytes(du.used)} used / {fmt_bytes(du.total)} total "
            f"({du.percent:.1f}%), {fmt_bytes(du.free)} free"
        )

    if metric in ("process", "proc", "self", "all"):
        p = psutil.Process()
        with p.oneshot():
            mem = p.memory_info()
            try:
                mem_pct = p.memory_percent()
            except Exception:
                mem_pct = 0.0
            cpu_p = p.cpu_percent(interval=0.3)
            threads = p.num_threads()
        lines.append(
            f"This process (PID {p.pid}, the LLM itself):\n"
            f"  RSS (physical RAM):  {fmt_bytes(mem.rss)}  ({mem_pct:.1f}% of system)\n"
            f"  CPU:                 {cpu_p:.1f}%\n"
            f"  Threads:             {threads}\n"
            f"  NOTE: RSS is the ONLY reliable number here. On macOS, "
            f"Activity Monitor's 'Memory' column shows 'phys_footprint' "
            f"which includes compressed memory and mmap'd model weights — "
            f"this value is NOT exposed by psutil and will typically be "
            f"1-3 GB HIGHER than RSS for an LLM process. If the user says "
            f"Activity Monitor shows a different number, that is expected "
            f"and NOT a contradiction — explain the gap honestly."
        )

    if metric in ("gpu", "ane", "all"):
        lines.append(
            "GPU/ANE: not directly queryable without sudo on Apple Silicon "
            "(requires `sudo powermetrics --samplers gpu_power,ane_power`)"
        )

    if metric in ("top", "procs", "processes", "all"):
        procs = list(psutil.process_iter(["pid", "name"]))
        for p in procs:
            try:
                p.cpu_percent(None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        time.sleep(0.3)
        snapshot = []
        for p in procs:
            try:
                with p.oneshot():
                    cpu_p = p.cpu_percent(None)
                    rss = p.memory_info().rss
                    name = p.info.get("name") or "?"
                    pid = p.info.get("pid")
                snapshot.append((cpu_p, rss, name, pid))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        my_pid = os.getpid()
        top_cpu = sorted(snapshot, key=lambda x: x[0], reverse=True)[:5]
        top_mem = sorted(snapshot, key=lambda x: x[1], reverse=True)[:5]
        lines.append("Top 5 by CPU:")
        for cpu_p, rss, name, pid in top_cpu:
            marker = " ← this is me (the LLM)" if pid == my_pid else ""
            lines.append(f"  {cpu_p:6.1f}%  {fmt_bytes(rss):>9}  {name} (pid {pid}){marker}")
        lines.append("Top 5 by RAM:")
        for cpu_p, rss, name, pid in top_mem:
            marker = " ← this is me (the LLM)" if pid == my_pid else ""
            lines.append(f"  {fmt_bytes(rss):>9}  {cpu_p:6.1f}%  {name} (pid {pid}){marker}")

    if metric in ("temp", "temperature", "all"):
        try:
            r = subprocess.run(
                ["smctemp", "-c"], capture_output=True, text=True, timeout=3
            )
            val = r.stdout.strip()
            try:
                t = float(val)
                if t > 0:
                    lines.append(f"CPU temperature: {t:.1f}°C")
                else:
                    lines.append("CPU temperature: sensor returned 0 (transient SMC read, try again)")
            except ValueError:
                lines.append(f"CPU temperature: unexpected output from smctemp: {val!r}")
        except FileNotFoundError:
            lines.append(
                "Temperature: smctemp not installed. "
                "Run `brew tap narugit/tap && brew install narugit/tap/smctemp` to enable."
            )
        except subprocess.TimeoutExpired:
            lines.append("Temperature: smctemp timed out")
        except Exception as e:
            lines.append(f"Temperature: error reading sensor ({e})")

    if not lines:
        return f"Unknown metric '{metric}'. Try: cpu, ram, disk, process, top, gpu, temp, all"

    return "\n".join(lines)
