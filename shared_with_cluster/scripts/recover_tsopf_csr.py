import csv
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "src" / "spmv.exe"
RAW = ROOT / "results" / "raw" / "team5_suitesparse_raw.csv"

MATRIX = ROOT / "datasets" / "original" / "TSOPF_FS_b39_c19.mtx"
VECTOR = ROOT / "datasets" / "vectors" / "TSOPF_FS_b39_c19_x.txt"

CONFIGS = [
    ("seq", "static", 1),
    ("omp", "static", 1),
    ("omp", "static", 4),
    ("omp", "static", 8),
    ("omp", "dynamic", 1),
    ("omp", "dynamic", 4),
    ("omp", "dynamic", 8),
    ("omp", "guided", 1),
    ("omp", "guided", 4),
    ("omp", "guided", 8),
]

if not RAW.exists():
    raise SystemExit("Main CSV not found. Stopping.")

with RAW.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    existing = list(reader)
    fields = reader.fieldnames

target = "TSOPF_FS_b39_c19"

present = {
    (r["implementation"], int(r["threads_or_ranks"]), r["schedule"])
    for r in existing
    if r["dataset"] == target and r["implementation"].startswith("csr_")
}

expected = {
    (f"csr_{mode}", threads, schedule)
    for mode, schedule, threads in CONFIGS
}

missing = expected - present

if not missing:
    print("All CSR configurations already exist. Nothing to recover.")
    raise SystemExit(0)

print("Existing CSR configurations:", len(present))
print("Missing CSR configurations:", len(missing))

env = os.environ.copy()
env["OMP_PROC_BIND"] = "false"
env["OMP_PLACES"] = "cores"

for mode, schedule, threads in CONFIGS:
    key = (f"csr_{mode}", threads, schedule)

    if key not in missing:
        continue

    print("\nRECOVERING:", key)

    cmd = [
        str(EXE),
        "--matrix", str(MATRIX),
        "--vector", str(VECTOR),
        "--dataset", target,
        "--format", "csr",
        "--mode", mode,
        "--schedule", schedule,
        "--threads", str(threads),
        "--chunk", "16" if mode == "omp" else "1",
        "--runs", "6",
        "--repetitions", "100",
        "--raw", str(RAW),
    ]

    subprocess.run(cmd, cwd=ROOT, env=env, check=True)

print("\nRecovery completed.")
