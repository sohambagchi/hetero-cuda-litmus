#!/usr/bin/env python3
"""
analyze.py — Result analysis for heterogeneous CPU-GPU litmus testing

Reads the results/ directory produced by tune.sh and generates summary tables.
Each subdirectory in results/ is named:
    test-tb-het_split-fence_scope-variant-membackend/
and contains:
    params.txt  — stress params that produced the best weak behavior rate
    rate        — best weak behavior rate (per second)
    weak        — weak behavior count at best rate
    total       — total behavior count at best rate

Usage:
    python3 analyze.py [results_dir] [--csv output.csv] [--log logfile]

If --log is provided, analyze raw tune.sh output (piped to a file) instead of
the results/ directory.
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_config_name(dirname):
    """Parse a result directory name into its components.

    Format: test-tb-het_split-fence_scope-variant-membackend

    Returns dict with keys: test, tb, het, fence_scope, variant, mem_backend, full_name
    """
    # The name has 6 dash-separated segments, but test names and TB names can
    # contain dashes/dots/plus (e.g., "2+2w", "read-rel-sys-and-cta",
    # "TB_01_2_3"). We parse from the right since the last 3 fields
    # (fence_scope, variant, mem_backend) are well-structured.
    parts = dirname.split("-")

    # Memory backend is always the last part
    mem_backend = parts[-1]

    # Variant is second-to-last
    variant = parts[-2]

    # Fence scope is third-to-last
    fence_scope = parts[-3]

    # HET split starts with "HET_" — find it by scanning from the right
    het_idx = None
    for i in range(len(parts) - 4, -1, -1):
        if parts[i].startswith("HET_"):
            het_idx = i
            break

    if het_idx is None:
        return None

    het = parts[het_idx]

    # TB starts with "TB_" — find it by scanning backwards from het_idx
    tb_idx = None
    for i in range(het_idx - 1, -1, -1):
        if parts[i].startswith("TB_"):
            tb_idx = i
            break

    if tb_idx is None:
        return None

    tb = "-".join(parts[tb_idx:het_idx])

    # Everything before tb_idx is the test name
    test = "-".join(parts[:tb_idx])

    return {
        "test": test,
        "tb": tb,
        "het": het,
        "fence_scope": fence_scope,
        "variant": variant,
        "mem_backend": mem_backend,
        "full_name": dirname,
    }


def read_result_dir(result_dir):
    """Read rate/weak/total from a result directory."""
    info = {}
    rate_file = os.path.join(result_dir, "rate")
    weak_file = os.path.join(result_dir, "weak")
    total_file = os.path.join(result_dir, "total")
    params_file = os.path.join(result_dir, "params.txt")

    if os.path.exists(rate_file):
        with open(rate_file) as f:
            try:
                info["rate"] = float(f.read().strip())
            except ValueError:
                info["rate"] = 0.0
    else:
        info["rate"] = 0.0

    if os.path.exists(weak_file):
        with open(weak_file) as f:
            try:
                info["weak"] = int(f.read().strip())
            except ValueError:
                info["weak"] = 0
    else:
        info["weak"] = 0

    if os.path.exists(total_file):
        with open(total_file) as f:
            try:
                info["total"] = int(f.read().strip())
            except ValueError:
                info["total"] = 0
    else:
        info["total"] = 0

    info["has_params"] = os.path.exists(params_file)
    return info


def analyze_results_dir(results_dir):
    """Analyze the results/ directory and print summary tables."""
    if not os.path.isdir(results_dir):
        print(f"Results directory '{results_dir}' not found.")
        return []

    entries = []
    for name in sorted(os.listdir(results_dir)):
        full_path = os.path.join(results_dir, name)
        if not os.path.isdir(full_path):
            continue
        config = parse_config_name(name)
        if config is None:
            print(f"  WARNING: Could not parse directory name: {name}")
            continue
        info = read_result_dir(full_path)
        config.update(info)
        entries.append(config)

    if not entries:
        print("No results found.")
        return []

    # ---- Summary by test ----
    print("=" * 80)
    print("SUMMARY BY TEST")
    print("=" * 80)

    by_test = defaultdict(list)
    for e in entries:
        by_test[e["test"]].append(e)

    print(f"{'Test':<25} {'Configs':>8} {'Max Rate':>12} {'Best Config':<40}")
    print("-" * 85)

    for test in sorted(by_test.keys()):
        configs = by_test[test]
        best = max(configs, key=lambda x: x["rate"])
        best_label = (
            f"{best['het']}/{best['tb']}/{best['fence_scope']}/{best['variant']}"
        )
        print(f"{test:<25} {len(configs):>8} {best['rate']:>12.1f} {best_label:<40}")

    print()

    # ---- Summary by HET split ----
    print("=" * 80)
    print("SUMMARY BY HET SPLIT")
    print("=" * 80)

    by_het = defaultdict(list)
    for e in entries:
        by_het[e["het"]].append(e)

    print(
        f"{'HET Split':<25} {'Configs':>8} {'Weak Configs':>12} {'Avg Rate':>12} {'Max Rate':>12}"
    )
    print("-" * 70)

    for het in sorted(by_het.keys()):
        configs = by_het[het]
        weak_configs = [c for c in configs if c["rate"] > 0]
        avg_rate = sum(c["rate"] for c in configs) / len(configs) if configs else 0
        max_rate = max(c["rate"] for c in configs) if configs else 0
        print(
            f"{het:<25} {len(configs):>8} {len(weak_configs):>12} {avg_rate:>12.1f} {max_rate:>12.1f}"
        )

    print()

    # ---- Summary by variant ----
    print("=" * 80)
    print("SUMMARY BY VARIANT")
    print("=" * 80)

    by_variant = defaultdict(list)
    for e in entries:
        key = e["variant"]
        if e["fence_scope"] != "NO_FENCE":
            key = f"{e['fence_scope']}/{e['variant']}"
        by_variant[key].append(e)

    print(f"{'Variant':<40} {'Configs':>8} {'Weak Configs':>12} {'Max Rate':>12}")
    print("-" * 72)

    for variant in sorted(by_variant.keys()):
        configs = by_variant[variant]
        weak_configs = [c for c in configs if c["rate"] > 0]
        max_rate = max(c["rate"] for c in configs) if configs else 0
        print(
            f"{variant:<40} {len(configs):>8} {len(weak_configs):>12} {max_rate:>12.1f}"
        )

    print()

    # ---- Summary by memory backend ----
    print("=" * 80)
    print("SUMMARY BY MEMORY BACKEND")
    print("=" * 80)

    by_mem = defaultdict(list)
    for e in entries:
        by_mem[e["mem_backend"]].append(e)

    print(
        f"{'Backend':<15} {'Configs':>8} {'Weak Configs':>12} {'Avg Rate':>12} {'Max Rate':>12}"
    )
    print("-" * 60)

    for mem in sorted(by_mem.keys()):
        configs = by_mem[mem]
        weak_configs = [c for c in configs if c["rate"] > 0]
        avg_rate = sum(c["rate"] for c in configs) / len(configs) if configs else 0
        max_rate = max(c["rate"] for c in configs) if configs else 0
        print(
            f"{mem:<15} {len(configs):>8} {len(weak_configs):>12} {avg_rate:>12.1f} {max_rate:>12.1f}"
        )

    print()

    # ---- Top 20 configurations by weak behavior rate ----
    print("=" * 80)
    print("TOP 20 CONFIGURATIONS BY WEAK BEHAVIOR RATE")
    print("=" * 80)

    sorted_entries = sorted(entries, key=lambda x: x["rate"], reverse=True)
    print(f"{'Configuration':<60} {'Rate':>12} {'Weak':>8} {'Total':>10}")
    print("-" * 90)

    for e in sorted_entries[:20]:
        label = f"{e['test']}/{e['het']}/{e['tb']}/{e['fence_scope']}/{e['variant']}"
        print(f"{label:<60} {e['rate']:>12.1f} {e['weak']:>8} {e['total']:>10}")

    print()

    # ---- Expected vs. actual weak behaviors ----
    print("=" * 80)
    print("EXPECTED WEAK vs ACTUAL WEAK")
    print("=" * 80)
    print("Configs with RELAXED or NO_FENCE variants are expected to potentially")
    print("show weak behaviors. Configs with stronger orderings should not.")
    print()

    expected_weak = []
    unexpected_weak = []
    expected_non_weak = []
    unexpected_non_weak = []

    for e in entries:
        is_relaxed = e["variant"] in ("RELAXED", "DEFAULT", "STORE_SC", "STORES_SC")
        has_weak = e["rate"] > 0

        if is_relaxed and has_weak:
            expected_weak.append(e)
        elif is_relaxed and not has_weak:
            expected_non_weak.append(e)  # relaxed but no weak seen (yet)
        elif not is_relaxed and has_weak:
            unexpected_weak.append(e)
        else:
            expected_non_weak.append(e)

    print(f"Expected weak (relaxed/default + weak seen):   {len(expected_weak)}")
    print(f"Unexpected weak (strong ordering + weak seen): {len(unexpected_weak)}")
    print(f"Expected non-weak (strong ordering, no weak):  {len(expected_non_weak)}")

    if unexpected_weak:
        print()
        print("UNEXPECTED WEAK BEHAVIORS:")
        for e in sorted(unexpected_weak, key=lambda x: x["rate"], reverse=True):
            print(f"  {e['full_name']}  rate={e['rate']:.1f}")

    print()

    return entries


def analyze_log_file(log_path):
    """Analyze raw tune.sh output (piped to file) — similar to original analyze.py."""
    if not os.path.exists(log_path):
        print(f"Log file '{log_path}' not found.")
        return

    test_stats = defaultdict(
        lambda: {"weak": 0, "total": 0, "max_rate": 0.0, "runs": 0}
    )
    total_iterations = 0

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Iteration:"):
                total_iterations += 1
                continue

            # Parse lines like:
            # test-tb-het-fence-variant-mem  weak: N, total: M, rate: R/s
            match = re.match(
                r"\s*(\S+)\s+weak:\s*(\d+),\s*total:\s*(\d+),\s*rate:\s*([\d.]+)/s",
                line,
            )
            if match:
                name = match.group(1)
                weak = int(match.group(2))
                total = int(match.group(3))
                rate = float(match.group(4))

                stats = test_stats[name]
                stats["weak"] += weak
                stats["total"] += total
                stats["runs"] += 1
                if rate > stats["max_rate"]:
                    stats["max_rate"] = rate

    print(f"Total iterations: {total_iterations}")
    print(f"Unique test configurations: {len(test_stats)}")
    print()

    print(
        f"{'Configuration':<60} {'Runs':>6} {'Weak':>10} {'Total':>12} {'MaxRate':>12}"
    )
    print("-" * 100)

    for name in sorted(
        test_stats.keys(), key=lambda k: test_stats[k]["max_rate"], reverse=True
    ):
        s = test_stats[name]
        print(
            f"{name:<60} {s['runs']:>6} {s['weak']:>10} {s['total']:>12} {s['max_rate']:>12.1f}"
        )


def export_csv(entries, csv_path):
    """Export results to CSV."""
    with open(csv_path, "w") as f:
        f.write("test,tb,het_split,fence_scope,variant,mem_backend,rate,weak,total\n")
        for e in sorted(entries, key=lambda x: (x["test"], x["het"], x["tb"])):
            f.write(
                f"{e['test']},{e['tb']},{e['het']},{e['fence_scope']},"
                f"{e['variant']},{e['mem_backend']},{e['rate']},{e['weak']},{e['total']}\n"
            )
    print(f"Results exported to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze heterogeneous litmus test results"
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="results",
        help="Path to results/ directory (default: results)",
    )
    parser.add_argument(
        "--csv",
        metavar="FILE",
        help="Export results to CSV file",
    )
    parser.add_argument(
        "--log",
        metavar="FILE",
        help="Analyze raw tune.sh log output instead of results/ directory",
    )

    args = parser.parse_args()

    if args.log:
        analyze_log_file(args.log)
    else:
        entries = analyze_results_dir(args.results_dir)
        if entries and args.csv:
            export_csv(entries, args.csv)


if __name__ == "__main__":
    main()
