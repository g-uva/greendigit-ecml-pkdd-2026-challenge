"""
Prepare Task B simulator inputs from the anonymized real-data summaries.

The challenge bundle does not include raw per-job events.  This script turns
the 15-minute site/VO aggregate rows into aggregate jobs, one per
site/VO/bucket, and writes simulator-ready files under data/.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


DEFAULT_RAW_METRICS = Path(__file__).resolve().parent / "raw_metrics"
DEFAULT_OUTPUT = Path(__file__).resolve().parent


def parse_bucket(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def fmt_dt(value: datetime) -> str:
    return value.isoformat()


def prepare(raw_metrics_dir: Path, output_dir: Path) -> None:
    source = raw_metrics_dir / "summary_sites_15m.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    by_site_bucket = defaultdict(lambda: {
        "energy_wh": 0.0,
        "cfp_g": 0.0,
        "records": 0.0,
        "ncores": 0.0,
    })
    site_totals = defaultdict(lambda: {
        "energy_wh": 0.0,
        "cfp_g": 0.0,
        "records": 0.0,
        "ncores": 0.0,
        "buckets": 0,
    })

    with source.open(newline="") as fh:
        for row in csv.DictReader(fh):
            bucket = parse_bucket(row["bucket_15m"])
            site_id = row["site_id"]
            records = float(row["records"])
            ncores = float(row["ncores"])
            energy_wh = float(row["energy_wh"])
            cfp_g = float(row["cfp_g"])

            rows.append({
                "bucket": bucket,
                "site_id": site_id,
                "vo": row["vo"],
                "records": records,
                "energy_wh": energy_wh,
                "cfp_g": cfp_g,
                "ncores": ncores,
            })

            agg = by_site_bucket[(site_id, bucket)]
            agg["energy_wh"] += energy_wh
            agg["cfp_g"] += cfp_g
            agg["records"] += records
            agg["ncores"] += ncores

    for (site_id, _bucket), agg in by_site_bucket.items():
        totals = site_totals[site_id]
        totals["energy_wh"] += agg["energy_wh"]
        totals["cfp_g"] += agg["cfp_g"]
        totals["records"] += agg["records"]
        totals["ncores"] += agg["ncores"]
        totals["buckets"] += 1

    write_job_trace(rows, output_dir / "job_trace.csv")
    write_forecast(by_site_bucket, output_dir / "forecast_baseline.csv")
    write_site_config(by_site_bucket, site_totals, output_dir / "site_config.json")


def write_job_trace(rows: list[dict], path: Path) -> None:
    fields = [
        "job_id", "series_id", "arrival_time", "deadline", "cpu_minutes",
        "memory_gb", "site_whitelist", "priority", "ncores", "records",
        "source_bucket_15m", "source_vo",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(sorted(rows, key=lambda r: (
                r["bucket"], r["site_id"], r["vo"]))):
            bucket = row["bucket"]
            site_id = row["site_id"]
            ncores = max(1, int(math.ceil(row["ncores"])))
            writer.writerow({
                "job_id": f"real-{idx:06d}",
                "series_id": site_id,
                "arrival_time": fmt_dt(bucket),
                "deadline": fmt_dt(bucket + timedelta(hours=24)),
                "cpu_minutes": 15.0,
                "memory_gb": max(1.0, round(ncores * 0.25, 2)),
                "site_whitelist": site_id,
                "priority": "normal",
                "ncores": ncores,
                "records": int(row["records"]),
                "source_bucket_15m": fmt_dt(bucket),
                "source_vo": row["vo"],
            })


def write_forecast(by_site_bucket: dict, path: Path) -> None:
    fields = [
        "series_id", "forecast_timestamp_utc", "horizon_steps_15m",
        "energy_wh_pred", "cfp_g_pred",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for (site_id, bucket), agg in sorted(by_site_bucket.items(),
                                            key=lambda item: (item[0][1], item[0][0])):
            writer.writerow({
                "series_id": site_id,
                "forecast_timestamp_utc": fmt_dt(bucket),
                "horizon_steps_15m": 4,
                "energy_wh_pred": round(agg["energy_wh"], 8),
                "cfp_g_pred": round(agg["cfp_g"], 8),
            })


def write_site_config(by_site_bucket: dict, site_totals: dict, path: Path) -> None:
    max_cores = defaultdict(float)
    max_records = defaultdict(float)
    for (site_id, _bucket), agg in by_site_bucket.items():
        max_cores[site_id] = max(max_cores[site_id], agg["ncores"])
        max_records[site_id] = max(max_records[site_id], agg["records"])

    sites = []
    for site_id in sorted(site_totals):
        totals = site_totals[site_id]
        avg_energy = totals["energy_wh"] / max(1, totals["buckets"])
        avg_carbon = totals["cfp_g"] / max(1, totals["buckets"])
        sites.append({
            "site_id": site_id,
            "name": f"Anonymized site {site_id}",
            "location": "ANON",
            "pue": 1.0,
            "dispatch_latency": 0.0,
            "capacity": {
                "max_cores": int(math.ceil(max_cores[site_id])),
                "max_memory_gb": round(max_cores[site_id] * 0.25, 2),
                "max_concurrent_jobs": int(math.ceil(max_records[site_id])),
            },
            "metadata": {
                "avg_energy_wh_per_15m": avg_energy,
                "avg_cfp_g_per_15m": avg_carbon,
                "source": "data/raw_metrics/summary_sites_15m.csv",
            },
        })

    with path.open("w") as fh:
        json.dump({"sites": sites}, fh, indent=2)
        fh.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-metrics-dir", type=Path, default=DEFAULT_RAW_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare(args.raw_metrics_dir, args.output_dir)


if __name__ == "__main__":
    main()
