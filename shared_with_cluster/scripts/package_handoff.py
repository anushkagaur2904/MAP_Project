import csv
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SH = ROOT / "shared_with_cluster"

def copy_file(src, dst):
    src, dst = Path(src), Path(dst)
    if src.is_file():
        if src.resolve() == dst.resolve():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

def copy_dir(src, dst):
    src, dst = Path(src), Path(dst)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)

# Selected datasets and reproducibility metadata.
copy_dir(ROOT / "datasets/original", SH / "datasets/original")
copy_dir(ROOT / "datasets/vectors", SH / "datasets/vectors")
copy_dir(ROOT / "datasets/metadata", SH / "datasets/metadata")

# Only reference outputs for the selected SuiteSparse matrices.
ref_src = ROOT / "shared_with_cluster/reference_outputs"
ref_dst = SH / "reference_outputs"
ref_dst.mkdir(parents=True, exist_ok=True)
for p in ref_src.glob("*_y_reference.txt"):
    if p.name not in {"test_small_y_reference.txt", "test_y.txt"}:
        copy_file(p, ref_dst / p.name)

# Source code, build instructions, configuration and scripts.
copy_file(ROOT / "README.md", SH / "BUILD_AND_METHOD.md")
copy_dir(ROOT / "config", SH / "config")
copy_dir(ROOT / "scripts", SH / "scripts")
copy_file(ROOT / "src/spmv.c", SH / "src/spmv.c")
copy_file(ROOT / "src/first_touch.c", SH / "src/first_touch.c")

# Hardware, STREAM measurements and report.
copy_dir(ROOT / "benchmarks", SH / "benchmarks")
copy_dir(ROOT / "report", SH / "report")

# Only the intended experiment CSVs; omit contaminated backups and pilots.
raw_dst = SH / "results/raw"
raw_dst.mkdir(parents=True, exist_ok=True)
for name in [
    "team5_suitesparse_raw.csv",
    "first_touch.csv",
    "thread_pinning.csv",
]:
    copy_file(ROOT / "results/raw" / name, raw_dst / name)

copy_dir(ROOT / "results/processed", SH / "results/processed")
copy_dir(ROOT / "results/plots", SH / "results/plots")

# Handoff note.
handoff = SH / "HANDOFF.md"
handoff.write_text("""# Team 5 – Cluster B SpMV Handoff

This package contains the CPU/OpenMP SpMV implementation and experiment results.

## Contents
- `datasets/original/`: selected SuiteSparse Matrix Market inputs
- `datasets/vectors/`: deterministic input vectors
- `datasets/metadata/`: matrix metadata and partition statistics
- `reference_outputs/`: reference output vectors
- `src/`: SpMV and first-touch C source
- `scripts/`, `config/`: experiment scripts and configuration
- `results/raw/`: benchmark, first-touch and thread-binding CSVs
- `results/processed/`: processed summaries
- `results/plots/`: generated figures
- `benchmarks/`: hardware and STREAM records
- `report/`: report draft, template and checklist

## Reproduction
Use MSYS2 UCRT64 with GCC and OpenMP. Build the SpMV program with:

`gcc -O3 -fopenmp -Wall -Wextra src/spmv.c -o src/spmv.exe -lm`

See `BUILD_AND_METHOD.md` and the files in `config/` and `scripts/` for experiment details.

## Important limitations
- STREAM Triad measured 12.7907 GB/s at 6 threads.
- NUMA locality was not verified.
- Thread affinity was attempted, but this libgomp configuration reported affinity unsupported.
- The roofline figure uses the measured bandwidth ceiling only; a compute ceiling was not established.
- Four matrices were excluded from ELL timing because their padding ratio exceeded the configured limit.
- Power measurements were not collected.

Raw benchmark CSV includes warm-up rows; use the `warmup` field when processing measurements.
""", encoding="utf-8")

# Generate SHA-256 manifest after all files are copied.
manifest = SH / "SHA256SUMS.csv"
with manifest.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["relative_path", "sha256", "bytes"])
    for p in sorted(SH.rglob("*")):
        if p.is_file() and p != manifest:
            writer.writerow([
                p.relative_to(SH).as_posix(),
                hashlib.sha256(p.read_bytes()).hexdigest(),
                p.stat().st_size,
            ])

print(f"Package prepared: {SH}")
print(f"Files included: {sum(1 for p in SH.rglob('*') if p.is_file())}")
print(f"Checksum manifest: {manifest}")
