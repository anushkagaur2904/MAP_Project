import csv
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "src/spmv.exe"
OUT = ROOT / "results/raw/thread_pinning.csv"

MATRICES = [
    ("sparsine", "sparsine"),
    ("TSOPF_FS_b39_c19", "TSOPF_FS_b39_c19"),
]
THREADS = [1, 4, 8]
BIND_VALUES = ["false", "true"]

if not EXE.exists():
    raise SystemExit(f"Executable not found: {EXE}")

OUT.parent.mkdir(parents=True, exist_ok=True)

if OUT.exists():
    raise SystemExit(f"{OUT} already exists. Rename it before rerunning.")

for dataset, stem in MATRICES:
    matrix = ROOT / f"datasets/original/{stem}.mtx"
    vector = ROOT / f"datasets/vectors/{stem}_x.txt"

    if not matrix.exists() or not vector.exists():
        raise SystemExit(f"Missing input for {dataset}")

    for bind in BIND_VALUES:
        for nt in THREADS:
            env = os.environ.copy()
            env["OMP_PROC_BIND"] = bind
            env["OMP_PLACES"] = "cores"

            cmd = [
                str(EXE),
                "--matrix", str(matrix),
                "--vector", str(vector),
                "--dataset", dataset,
                "--format", "csr",
                "--mode", "omp",
                "--schedule", "static",
                "--threads", str(nt),
                "--chunk", "16",
                "--runs", "6",
                "--repetitions", "100",
                "--raw", str(OUT),
            ]

            print(
                f"Running {dataset}: threads={nt}, "
                f"OMP_PROC_BIND={bind}",
                flush=True,
            )

            subprocess.run(
                cmd,
                cwd=ROOT,
                env=env,
                check=True,
                timeout=3600,
            )

print(f"Thread pinning experiment completed: {OUT}")
