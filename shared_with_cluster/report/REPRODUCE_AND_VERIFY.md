# Team 5 – Reproduction and Verification Guide
## For a teammate or a new person checking the SpMV project

This guide explains how to inspect the completed project, verify its files and results, regenerate analysis plots, and deliberately rerun experiments.

**Important:** Checking the completed results does not require rerunning the full benchmark. Start with the verification steps. Do not run the full benchmark over the existing raw CSV without backing it up.

## 1. What is in this project?

| Path | Purpose |
|---|---|
| `src/spmv.c` | COO, CSR and ELL SpMV implementation |
| `src/first_touch.c` | First-touch initialization experiment |
| `config/` | Matrix and experiment configuration |
| `datasets/original/` | SuiteSparse Matrix Market inputs |
| `datasets/vectors/` | Saved input vectors |
| `datasets/metadata/` | Matrix dimensions and row statistics |
| `shared_with_cluster/reference_outputs/` | Reference output vectors in the original project |
| `results/raw/` | Raw benchmark and additional experiment CSVs |
| `results/processed/` | Processed performance summaries |
| `results/plots/` | Generated figures |
| `benchmarks/` | Hardware and STREAM records |
| `report/TEAM5_REPORT.md` | Final report |
| `scripts/` | Preparation, benchmark, analysis and validation scripts |

When working from the shared handoff folder, the reference outputs are in `reference_outputs/`, and the report is in `report/`.

## 2. Required environment

The tested environment was Windows 11 with MSYS2 UCRT64, GCC 16.2.0 and OpenMP.

Install or make available:
- MSYS2 UCRT64 terminal.
- GCC with OpenMP support.
- Python 3.
- Python packages: NumPy, pandas and Matplotlib.

Open the **MSYS2 UCRT64** terminal, not the MINGW64 terminal.

Move to the project directory:

`cd /c/Users/<your-user>/<project-folder>`

The path above is an example. Use the actual location where the project was extracted.

Check the compiler:

`gcc --version`

Check Python:

`python --version`

Check that the compiler accepts OpenMP:

`echo | gcc -fopenmp -dM -E - | grep _OPENMP`

The tested environment returned `_OPENMP 202111`.

## 3. Build the SpMV program

From the project root, compile:

`gcc -O3 -fopenmp -Wall -Wextra src/spmv.c -o src/spmv.exe -lm`

The source may produce harmless unused-function warnings. A successful compile should create `src/spmv.exe`.

The implementation's command-line options and benchmark usage are documented in `README.md` (or `BUILD_AND_METHOD.md` in the shared package). Read that file before running a benchmark.

## 4. Verify the existing package

### A. Check the benchmark CSV

The main raw file is:

`results/raw/team5_suitesparse_raw.csv`

The expected file contains 2,100 data rows, excluding the header.

### B. Run the project validator

`python scripts/validate_results.py`

The expected successful output includes:

`Core CSV and measurement checks passed`

The validator checks core CSV and measurement consistency. It does not independently prove hardware specifications, STREAM values, plot interpretation or matrix categorization.

### C. Verify package checksums

From the shared package root, run:

`python - <<'PY'`
`import csv, hashlib`
`from pathlib import Path`
`root = Path('.')`
`rows = list(csv.DictReader((root / 'SHA256SUMS.csv').open(encoding='utf-8')))`
`bad = []`
`for row in rows:`
`    p = root / row['relative_path']`
`    if not p.is_file() or p.stat().st_size != int(row['bytes']):`
`        bad.append(row['relative_path'])`
`    elif hashlib.sha256(p.read_bytes()).hexdigest() != row['sha256']:`
`        bad.append(row['relative_path'])`
`print('Files checked:', len(rows))`
`print('Failures:', bad)`
`print('PASS' if not bad else 'FAIL')`
`PY`

Expected result for the packaged version: 112 files checked and no failures. If files are edited, the manifest must be regenerated before sharing.

## 5. Regenerate processed summaries and plots

From the project root, run:

`python scripts/analyze_results.py`

This reads the benchmark results and writes processed summaries and plots under `results/processed/` and `results/plots/`.

Check that the plots exist:

`ls results/plots`

The required graph categories are:
1. GFLOP/s by format and matrix.
2. Speedup versus thread count.
3. Scheduling comparison.
4. Load imbalance versus matrix irregularity.
5. Roofline analysis.

The roofline plot is bandwidth-only. Do not describe it as having a measured compute ceiling.

## 6. How to rerun experiments safely

The existing raw CSV is the completed result set. Preserve a backup before collecting new measurements.

The full benchmark runner can execute many matrix/format/thread/schedule configurations and may append or regenerate raw results. Do not run it casually against the final result file.

Before a fresh full experiment:
1. Make a separate copy of the project or create a separate results directory.
2. Back up `results/raw/team5_suitesparse_raw.csv`.
3. Review `config/experiment.json` and `config/matrices.csv`.
4. Check available disk space and allow sufficient runtime.
5. Run `python scripts/run_benchmarks.py` only when a complete fresh benchmark is intended.
6. Validate the resulting CSV with `python scripts/validate_results.py`.
7. Regenerate summaries and plots with `python scripts/analyze_results.py`.

For first-touch or thread-binding experiments, inspect the corresponding script before running it because it writes result files. Existing results are:
- `results/raw/first_touch.csv`
- `results/raw/thread_pinning.csv`

Thread binding may report that affinity is unsupported on the tested Windows/MSYS2 `libgomp` runtime. If that happens, do not claim that threads were successfully pinned.

## 7. How to interpret the results

- Compare formats on the same matrix and thread count before drawing conclusions.
- Use the per-matrix speedup plots rather than relying only on pooled averages.
- The scheduling comparison pools multiple configurations and is descriptive.
- The load-imbalance plot uses a deterministic static-partition estimate, not runtime instrumentation.
- ELL padding can make the format unsuitable for highly irregular row lengths.
- STREAM Triad bandwidth is measured memory bandwidth, not theoretical peak bandwidth.
- NUMA locality was not verified.
- No power measurements were collected.

## 8. Final completion checklist

- [ ] Project compiles successfully.
- [ ] All selected matrix and vector files are present.
- [ ] Reference outputs are present.
- [ ] Main raw CSV has the expected 2,100 data rows.
- [ ] `python scripts/validate_results.py` passes.
- [ ] `python scripts/analyze_results.py` completes.
- [ ] All required graph categories are present.
- [ ] Report is present at `report/TEAM5_REPORT.md`.
- [ ] Hardware and STREAM records are present.
- [ ] Limitations are understood and accurately described.
- [ ] SHA-256 verification passes after the final files are copied.

## 9. Where to start

If you only want to check whether the work is complete, follow Sections 3, 4, 5 and 8. You do not need to rerun all experiments.

If you need to understand what was done, read `report/PROJECT_WORK_LOG.md` and then `report/TEAM5_REPORT.md`.

If you need to reproduce the full experiment, read this guide and `README.md` first.
