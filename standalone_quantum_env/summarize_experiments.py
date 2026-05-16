#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


SUMMARY_FIELDS = [
    "success_rate",
    "route_completion_rate",
    "fusion_failure_rate",
    "mean_ghz_fidelity",
    "mean_total_time",
    "mean_total_hops",
    "mean_reward",
    "conversion_failure_rate",
    "memory_failure_rate",
    "mean_memory_wait_time",
    "mean_memory_decay_loss",
    "mean_photonic_route_attempts",
]


def add_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize RELiQ-ME experiment JSON files.")
    parser.add_argument(
        "--input-glob",
        default="output/experiments/model_effect_*.json",
        help="Glob of experiment JSON files, relative to standalone_quantum_env unless absolute.",
    )
    parser.add_argument("--csv", type=Path, default=Path("output/experiments/summary.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("output/experiments/summary.md"))
    return parser


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_backend(path: Path, payload: dict[str, Any]) -> str:
    stem = path.stem
    if "dynamic_default" in stem:
        return "dynamic_default"
    if "dynamic_perfect" in stem:
        return "dynamic_perfect"
    if "_routing_" in stem:
        return "routing_policy"
    if "_path_" in stem:
        return "path_estimator"
    if payload.get("bipartite_backend"):
        return str(payload["bipartite_backend"])
    return "unknown"


def infer_n_router(path: Path, payload: dict[str, Any]) -> str:
    metrics = payload.get("metrics") or []
    if metrics and "n_router" in metrics[0]:
        return str(metrics[0]["n_router"])
    for token in path.stem.split("_"):
        if token.startswith("n") and token[1:].isdigit():
            return token[1:]
    return ""


def rows_from_file(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    summary = payload.get("summary", {})
    backend = infer_backend(path, payload)
    n_router = infer_n_router(path, payload)
    rows = []
    for policy, values in sorted(summary.items()):
        row = {
            "file": path.name,
            "backend": backend,
            "n_router": n_router,
            "policy": policy,
            "episodes": values.get("episodes", 0),
        }
        for field in SUMMARY_FIELDS:
            row[field] = values.get(field, "")
        row["failure_reason_counts"] = json.dumps(values.get("failure_reason_counts", {}), sort_keys=True)
        rows.append(row)
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["backend"]), str(row["n_router"]), str(row["policy"])), []).append(row)

    aggregated = []
    for (backend, n_router, policy), group in sorted(groups.items()):
        out = {
            "backend": backend,
            "n_router": n_router,
            "policy": policy,
            "runs": len(group),
            "episodes": sum(int(row.get("episodes") or 0) for row in group),
        }
        for field in SUMMARY_FIELDS:
            values = [float(row[field]) for row in group if row.get(field) != ""]
            out[f"{field}_mean"] = mean(values) if values else ""
            out[f"{field}_std"] = pstdev(values) if len(values) > 1 else 0.0 if values else ""
        aggregated.append(out)
    return aggregated


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        "backend",
        "n_router",
        "policy",
        "runs",
        "episodes",
        "success_rate_mean",
        "mean_ghz_fidelity_mean",
        "mean_total_time_mean",
        "mean_total_hops_mean",
        "mean_reward_mean",
        "conversion_failure_rate_mean",
        "memory_failure_rate_mean",
        "mean_photonic_route_attempts_mean",
    ]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = add_args()
    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parent
    input_glob = Path(args.input_glob)
    if input_glob.is_absolute():
        paths = sorted(input_glob.parent.glob(input_glob.name))
    else:
        paths = sorted(base_dir.glob(args.input_glob))

    raw_rows = []
    for path in paths:
        raw_rows.extend(rows_from_file(path))
    aggregated = aggregate_rows(raw_rows)

    csv_path = args.csv if args.csv.is_absolute() else base_dir / args.csv
    markdown_path = args.markdown if args.markdown.is_absolute() else base_dir / args.markdown
    write_csv(csv_path, aggregated)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_table(aggregated), encoding="utf-8")
    print(f"input_files={len(paths)} rows={len(aggregated)} csv={csv_path} markdown={markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
