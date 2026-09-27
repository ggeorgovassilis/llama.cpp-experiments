#!/usr/bin/env python3
"""Summarise nvidia-smi dmon utilisation logs into per-GPU averages.

Usage: summarize_dmon.py <dmon.log>
"""
import sys


def main():
    path = sys.argv[1]
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # expected: gpu sm mem  (dmon -s u)  -> 3 columns
            try:
                gpu = int(parts[0])
                sm = int(parts[1])
            except (ValueError, IndexError):
                continue
            rows.append((gpu, sm))

    if not rows:
        print("no samples")
        return

    gpus = sorted({g for g, _ in rows})
    print(f"samples={len(rows)}")
    for g in gpus:
        vals = [sm for gg, sm in rows if gg == g]
        avg = sum(vals) / len(vals)
        mx = max(vals)
        busy = sum(1 for v in vals if v > 5)
        print(f"gpu{g}: avg_sm={avg:.1f}% max_sm={mx}% busy_frac={busy/len(vals):.2f}")


if __name__ == "__main__":
    main()
