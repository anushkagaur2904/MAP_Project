import csv
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cfg = json.loads((ROOT / "config/experiment.json").read_text())
exe = ROOT / "src/spmv.exe"

matrix = ROOT / "datasets/original/qpband.mtx"
vector = ROOT / "datasets/vectors/qpband_x.txt"
raw = ROOT / "results/raw/team5_suitesparse_raw.csv"

if not exe.exists():
    raise SystemExit("Executable not found.")

if not matrix.exists() or not vector.exists():
    raise SystemExit("qpband matrix or vector missing.")

if not raw.exists():
    raise SystemExit("Existing benchmark CSV not found.")

refdir = ROOT / "shared_with_cluster/reference_outputs"
refdir.mkdir(parents=True, exist_ok=True)
refpath = refdir / "qpband_y_reference.txt"

refcmd = [
    str(exe), "--matrix", str(matrix),
    "--vector", str(vector),
    "--dataset", "qpband",
    "--format", "csr", "--mode", "seq",
    "--threads", "1", "--runs", "2",
    "--reference-vector", str(refpath)
]

subprocess.run(refcmd, cwd=ROOT, check=True)

for fmt in cfg["formats"]:
    for mode in cfg["modes"]:
        schedules = cfg["schedules"] if mode == "omp" else ["static"]

        for schedule in schedules:
            threads_list = cfg["threads"] if mode == "omp" else [1]

            for threads in threads_list:
                chunks = cfg["chunks"] if mode == "omp" else [1]

                for chunk in chunks:
                    for bind in cfg.get("proc_bind", ["false"]):
                        env = os.environ.copy()
                        env["OMP_PROC_BIND"] = str(bind)
                        env["OMP_PLACES"] = "cores"

                        cmd = [
                            str(exe),
                            "--matrix", str(matrix),
                            "--vector", str(vector),
                            "--dataset", "qpband",
                            "--format", fmt,
                            "--mode", mode,
                            "--schedule", schedule,
                            "--threads", str(threads),
                            "--chunk", str(chunk),
                            "--runs", str(cfg["runs"]),
                            "--repetitions", str(cfg["repetitions"]),
                            "--raw", str(raw)
                        ]

                        print(
                            "RUN:", fmt, mode, schedule,
                            threads, "threads, bind =", bind,
                            flush=True
                        )

                        subprocess.run(
                            cmd, cwd=ROOT, env=env,
                            check=True,
                            timeout=cfg["timeout_seconds"]
                        )

print("qpband benchmark completed.")
