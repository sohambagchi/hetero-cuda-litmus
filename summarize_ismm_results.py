#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collapse ISMM results.csv into per-experiment weak percentages."
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default="full-matrix-results/ismm/results.csv",
        help="Path to ISMM results.csv",
    )
    parser.add_argument(
        "--csv",
        dest="output_csv",
        help="Optional output CSV path for collapsed results",
    )
    return parser.parse_args()


def load_rows(csv_path: Path):
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def collapse(rows):
    grouped = defaultdict(
        lambda: {
            "expected": "",
            "runs": 0,
            "ok_runs": 0,
            "run_failures": 0,
            "total_behaviors": 0,
            "weak_behaviors": 0,
            "runs_with_weak": 0,
            "max_weak_behaviors": 0,
        }
    )

    for row in rows:
        experiment = row["experiment"]
        status = row["status"]
        total_behaviors = int(row["total_behaviors"] or 0)
        weak_behaviors = int(row["weak_behaviors"] or 0)

        entry = grouped[experiment]
        entry["expected"] = row["expected"]
        entry["runs"] += 1

        if status == "ok":
            entry["ok_runs"] += 1
            entry["total_behaviors"] += total_behaviors
            entry["weak_behaviors"] += weak_behaviors
            if weak_behaviors > 0:
                entry["runs_with_weak"] += 1
            if weak_behaviors > entry["max_weak_behaviors"]:
                entry["max_weak_behaviors"] = weak_behaviors
        else:
            entry["run_failures"] += 1

    collapsed = []
    for experiment, entry in sorted(grouped.items()):
        total_behaviors = entry["total_behaviors"]
        weak_behaviors = entry["weak_behaviors"]
        weak_pct = (100.0 * weak_behaviors / total_behaviors) if total_behaviors else 0.0
        hit_rate_pct = (100.0 * entry["runs_with_weak"] / entry["ok_runs"]) if entry["ok_runs"] else 0.0
        collapsed.append(
            {
                "experiment": experiment,
                "expected": entry["expected"],
                "runs": entry["runs"],
                "ok_runs": entry["ok_runs"],
                "run_failures": entry["run_failures"],
                "total_behaviors": total_behaviors,
                "weak_behaviors": weak_behaviors,
                "weak_pct": weak_pct,
                "runs_with_weak": entry["runs_with_weak"],
                "hit_rate_pct": hit_rate_pct,
                "max_weak_behaviors": entry["max_weak_behaviors"],
            }
        )
    return collapsed


def print_table(collapsed):
    headers = (
        "experiment",
        "expected",
        "ok_runs",
        "weak_behaviors",
        "total_behaviors",
        "weak_pct",
        "runs_with_weak",
        "hit_rate_pct",
    )
    print(
        f"{'experiment':<34} {'expected':<11} {'ok_runs':>7} {'weak':>12} {'total':>12} {'weak_%':>9} {'hit_runs':>10} {'hit_%':>8}"
    )
    print("-" * 112)
    for row in collapsed:
        print(
            f"{row['experiment']:<34} {row['expected']:<11} {row['ok_runs']:>7} "
            f"{row['weak_behaviors']:>12} {row['total_behaviors']:>12} "
            f"{row['weak_pct']:>8.4f}% {row['runs_with_weak']:>10} {row['hit_rate_pct']:>7.2f}%"
        )


def write_csv(output_path: Path, collapsed):
    fieldnames = [
        "experiment",
        "expected",
        "runs",
        "ok_runs",
        "run_failures",
        "weak_behaviors",
        "total_behaviors",
        "weak_pct",
        "runs_with_weak",
        "hit_rate_pct",
        "max_weak_behaviors",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(collapsed)


def main():
    args = parse_args()
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        raise SystemExit(f"results file not found: {csv_path}")

    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit("results file is empty")

    collapsed = collapse(rows)
    print_table(collapsed)

    if args.output_csv:
        write_csv(Path(args.output_csv), collapsed)


if __name__ == "__main__":
    main()
