#!/usr/bin/env python3
"""
Team 7 — CUDA SpMV Publication-Quality Plotting Suite
Generates all 12 required benchmark visualization figures.
"""

import os
import sys
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = ROOT / "results" / "raw" / "team7_raw.csv"
SUMMARY_CSV = ROOT / "results" / "summary" / "team7_summary.csv"
METADATA_CSV = ROOT / "datasets" / "metadata" / "matrix_metadata.csv"
HW_JSON = ROOT / "results" / "hardware_info.json"

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

def load_data():
    if not SUMMARY_CSV.exists():
        print(f"Summary CSV not found: {SUMMARY_CSV}")
        return None, None, None
    summary_df = pd.read_csv(SUMMARY_CSV)
    raw_df = pd.read_csv(RAW_CSV) if RAW_CSV.exists() else None
    meta_df = pd.read_csv(METADATA_CSV) if METADATA_CSV.exists() else None
    return summary_df, raw_df, meta_df

def get_hardware_info():
    hw = {"gpu_name": "NVIDIA GPU", "peak_bandwidth_gbs": 320.0, "peak_fp64_gflops": 254.0}
    if HW_JSON.exists():
        try:
            with open(HW_JSON, "r") as f:
                data = json.load(f)
                hw.update(data)
        except Exception as e:
            print(f"Warning reading {HW_JSON}: {e}")
    return hw

def plot_1_execution_time_vs_size(summary_df):
    plt.figure(figsize=(9, 6))
    for impl, group in summary_df.groupby("implementation"):
        sorted_group = group.sort_values(by="nnz")
        plt.plot(sorted_group["nnz"], sorted_group["mean_kernel_ms"], marker="o", label=impl)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Nonzero Elements (NNZ)")
    plt.ylabel("Execution Time (ms)")
    plt.title("Execution Time vs. Matrix Size (NNZ)")
    plt.grid(True)
    plt.legend()
    plt.savefig(PLOTS_DIR / "1_execution_time_vs_size.png", dpi=300)
    plt.close()

def plot_2_gflops_comparison(summary_df):
    plt.figure(figsize=(12, 6))
    datasets = summary_df["dataset"].unique()
    impls = summary_df["implementation"].unique()
    x = np.arange(len(datasets))
    width = 0.8 / max(len(impls), 1)

    for i, impl in enumerate(impls):
        subset = summary_df[summary_df["implementation"] == impl]
        vals = []
        errs = []
        for d in datasets:
            row = subset[subset["dataset"] == d]
            if not row.empty:
                vals.append(row["mean_gflops"].values[0])
                # Approximate GFLOP/s error from kernel time std
                mean_t = row["mean_kernel_ms"].values[0]
                std_t = row["std_kernel_ms"].values[0]
                errs.append((std_t / mean_t) * row["mean_gflops"].values[0] if mean_t > 0 else 0)
            else:
                vals.append(0)
                errs.append(0)
        plt.bar(x + i * width, vals, width, yerr=errs, capsize=3, label=impl)

    plt.xlabel("Matrix Dataset")
    plt.ylabel("Throughput (GFLOP/s)")
    plt.title("GFLOP/s Comparison Across All SpMV Implementations")
    plt.xticks(x + width * (len(impls) - 1) / 2, datasets, rotation=45, ha="right")
    plt.grid(True, axis="y")
    plt.legend()
    plt.savefig(PLOTS_DIR / "2_gflops_comparison.png", dpi=300)
    plt.close()

def plot_3_speedup_vs_cpu(summary_df):
    plt.figure(figsize=(12, 6))
    cpu_subset = summary_df[summary_df["implementation"] == "cpu_sequential"]
    if cpu_subset.empty:
        return
    cpu_times = dict(zip(cpu_subset["dataset"], cpu_subset["mean_kernel_ms"]))

    datasets = summary_df["dataset"].unique()
    gpu_impls = [imp for imp in summary_df["implementation"].unique() if imp != "cpu_sequential"]
    if not gpu_impls:
        return

    x = np.arange(len(datasets))
    width = 0.8 / len(gpu_impls)

    for i, impl in enumerate(gpu_impls):
        subset = summary_df[summary_df["implementation"] == impl]
        speedups = []
        for d in datasets:
            row = subset[subset["dataset"] == d]
            cpu_t = cpu_times.get(d, 0)
            if not row.empty and row["mean_kernel_ms"].values[0] > 0 and cpu_t > 0:
                speedups.append(cpu_t / row["mean_kernel_ms"].values[0])
            else:
                speedups.append(0)
        plt.bar(x + i * width, speedups, width, label=impl)

    plt.axhline(1.0, color="red", linestyle="--", label="CPU Baseline (1.0x)")
    plt.xlabel("Matrix Dataset")
    plt.ylabel("Speedup vs. CPU Sequential")
    plt.title("GPU Acceleration: Speedup vs. Sequential CPU Baseline")
    plt.xticks(x + width * (len(gpu_impls) - 1) / 2, datasets, rotation=45, ha="right")
    plt.grid(True, axis="y")
    plt.legend()
    plt.savefig(PLOTS_DIR / "3_speedup_vs_cpu.png", dpi=300)
    plt.close()

def plot_4_speedup_vs_ideal(summary_df):
    plt.figure(figsize=(10, 6))
    # Comparison against memory bandwidth upper bound
    hw = get_hardware_info()
    datasets = summary_df["dataset"].unique()
    impls = summary_df["implementation"].unique()
    x = np.arange(len(datasets))
    width = 0.8 / max(len(impls), 1)

    for i, impl in enumerate(impls):
        subset = summary_df[summary_df["implementation"] == impl]
        bws = []
        for d in datasets:
            row = subset[subset["dataset"] == d]
            if not row.empty:
                bws.append(row["bandwidth_gbs"].values[0])
            else:
                bws.append(0)
        plt.bar(x + i * width, bws, width, label=impl)

    plt.axhline(hw["peak_bandwidth_gbs"], color="black", linestyle="--",
                label=f"Ideal Peak Bandwidth ({hw['peak_bandwidth_gbs']} GB/s)")
    plt.xlabel("Matrix Dataset")
    plt.ylabel("Effective Bandwidth (GB/s)")
    plt.title("Achieved Bandwidth vs. Ideal Hardware Peak")
    plt.xticks(x + width * (len(impls) - 1) / 2, datasets, rotation=45, ha="right")
    plt.grid(True, axis="y")
    plt.legend()
    plt.savefig(PLOTS_DIR / "4_speedup_vs_ideal.png", dpi=300)
    plt.close()

def plot_5_parallel_efficiency(raw_df):
    if raw_df is None or raw_df.empty:
        return
    # Examine block size scaling efficiency
    plt.figure(figsize=(9, 6))
    vec_data = raw_df[raw_df["implementation"] == "vector_csr"]
    if vec_data.empty:
        vec_data = raw_df
    grouped = vec_data.groupby(["dataset", "threads_or_ranks"])["kernel_ms"].mean().unstack()
    if not grouped.empty:
        grouped.plot(kind="bar", figsize=(11, 6))
        plt.xlabel("Matrix Dataset")
        plt.ylabel("Kernel Execution Time (ms)")
        plt.title("Thread Configuration & Warp Scaling Efficiency")
        plt.grid(True, axis="y")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Configuration")
        plt.savefig(PLOTS_DIR / "5_parallel_efficiency.png", dpi=300)
        plt.close()

def plot_6_roofline(summary_df):
    hw = get_hardware_info()
    peak_bw = hw["peak_bandwidth_gbs"]
    peak_flops = hw["peak_fp64_gflops"]

    # Operational Intensity for FP64 SpMV:
    # 2 FLOPs / (12 bytes for index+val + ~8-16 bytes vector traffic) approx 0.10 - 0.25 FLOP/byte
    intensities = np.logspace(-2, 2, 200)
    roofline = np.minimum(peak_bw * intensities, peak_flops)

    plt.figure(figsize=(9, 6))
    plt.loglog(intensities, roofline, "k-", linewidth=2, label=f"Roofline ({hw['gpu_name']})")
    plt.axhline(peak_flops, color="gray", linestyle=":", label=f"Peak FP64: {peak_flops} GFLOP/s")

    for impl, group in summary_df.groupby("implementation"):
        # Calculate operational intensity: FLOPs / Bytes
        # FLOPs = 2 * nnz, Bytes = (2 * nnz) / throughput * bandwidth
        oi = []
        perf = []
        for _, row in group.iterrows():
            if row["bandwidth_gbs"] > 0 and row["mean_gflops"] > 0:
                oi.append(row["mean_gflops"] / row["bandwidth_gbs"])
                perf.append(row["mean_gflops"])
        if oi:
            plt.scatter(oi, perf, s=50, alpha=0.8, label=impl)

    plt.xlabel("Operational Intensity (FLOP/Byte)")
    plt.ylabel("Performance (GFLOP/s)")
    plt.title(f"Roofline Analysis — {hw['gpu_name']}")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(PLOTS_DIR / "6_roofline_plot.png", dpi=300)
    plt.close()

def plot_7_stacked_time_breakdown(summary_df):
    plt.figure(figsize=(11, 6))
    # Select unique datasets for a representative implementation
    gpu_df = summary_df[summary_df["implementation"] != "cpu_sequential"]
    if gpu_df.empty:
        gpu_df = summary_df
    impl = gpu_df["implementation"].iloc[0]
    subset = gpu_df[gpu_df["implementation"] == impl].drop_duplicates(subset=["dataset"])

    datasets = subset["dataset"]
    kernel_times = subset["mean_kernel_ms"]
    transfer_times = subset["mean_transfer_ms"]

    x = np.arange(len(datasets))
    plt.bar(x, kernel_times, label="Kernel Compute Time")
    plt.bar(x, transfer_times, bottom=kernel_times, label="Memory Transfer (H2D + D2H)")

    plt.xlabel("Matrix Dataset")
    plt.ylabel("Time (ms)")
    plt.title(f"Time Breakdown: Compute vs. Data Transfer ({impl})")
    plt.xticks(x, datasets, rotation=45, ha="right")
    plt.grid(True, axis="y")
    plt.legend()
    plt.savefig(PLOTS_DIR / "7_stacked_time_breakdown.png", dpi=300)
    plt.close()

def plot_8_ell_width_vs_time(summary_df, meta_df):
    plt.figure(figsize=(9, 6))
    ell_data = summary_df[summary_df["implementation"] == "ell"]
    if not ell_data.empty and meta_df is not None:
        merged = pd.merge(ell_data, meta_df, left_on="dataset", right_on="matrix_id")
        plt.scatter(merged["ell_width"], merged["mean_kernel_ms"], c=merged["ell_padding_ratio"], 
                    cmap="viridis", s=80, edgecolors="black")
        cbar = plt.colorbar()
        cbar.set_label("ELL Padding Ratio")
        for _, row in merged.iterrows():
            plt.annotate(row["dataset"], (row["ell_width"], row["mean_kernel_ms"]),
                         fontsize=8, xytext=(4, 2), textcoords="offset points")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("ELL Width (Max Row Length)")
        plt.ylabel("Execution Time (ms)")
        plt.title("ELLPACK: ELL Width vs. Execution Time")
        plt.grid(True)
        plt.savefig(PLOTS_DIR / "8_ell_width_vs_time.png", dpi=300)
        plt.close()

def plot_9_hyb_width_vs_time(raw_df):
    if raw_df is None or raw_df.empty:
        return
    hyb_data = raw_df[raw_df["implementation"] == "hyb"]
    if not hyb_data.empty:
        plt.figure(figsize=(10, 6))
        grouped = hyb_data.groupby(["dataset", "threads_or_ranks"])["kernel_ms"].mean().unstack()
        if not grouped.empty:
            grouped.plot(kind="bar", figsize=(11, 6))
            plt.xlabel("Matrix Dataset")
            plt.ylabel("Execution Time (ms)")
            plt.title("HYB SpMV: ELL Width Tuning vs. Execution Time")
            plt.grid(True, axis="y")
            plt.xticks(rotation=45, ha="right")
            plt.legend(title="HYB Configuration")
            plt.savefig(PLOTS_DIR / "9_hyb_width_vs_time.png", dpi=300)
            plt.close()

def plot_10_achieved_bandwidth(summary_df):
    plt.figure(figsize=(12, 6))
    datasets = summary_df["dataset"].unique()
    impls = summary_df["implementation"].unique()
    x = np.arange(len(datasets))
    width = 0.8 / max(len(impls), 1)

    for i, impl in enumerate(impls):
        subset = summary_df[summary_df["implementation"] == impl]
        vals = []
        for d in datasets:
            row = subset[subset["dataset"] == d]
            vals.append(row["bandwidth_gbs"].values[0] if not row.empty else 0)
        plt.bar(x + i * width, vals, width, label=impl)

    plt.xlabel("Matrix Dataset")
    plt.ylabel("Achieved Memory Bandwidth (GB/s)")
    plt.title("Achieved Memory Bandwidth Per Implementation")
    plt.xticks(x + width * (len(impls) - 1) / 2, datasets, rotation=45, ha="right")
    plt.grid(True, axis="y")
    plt.legend()
    plt.savefig(PLOTS_DIR / "10_achieved_bandwidth.png", dpi=300)
    plt.close()

def plot_11_cusparse_percentage(summary_df):
    plt.figure(figsize=(12, 6))
    cusparse_subset = summary_df[summary_df["implementation"] == "cusparse"]
    if cusparse_subset.empty:
        return
    cusparse_gflops = dict(zip(cusparse_subset["dataset"], cusparse_subset["mean_gflops"]))

    custom_impls = [imp for imp in summary_df["implementation"].unique() 
                    if imp not in ["cusparse", "cpu_sequential"]]
    if not custom_impls:
        return

    datasets = summary_df["dataset"].unique()
    x = np.arange(len(datasets))
    width = 0.8 / len(custom_impls)

    for i, impl in enumerate(custom_impls):
        subset = summary_df[summary_df["implementation"] == impl]
        pcts = []
        for d in datasets:
            row = subset[subset["dataset"] == d]
            cg = cusparse_gflops.get(d, 0)
            if not row.empty and cg > 0:
                # Custom GFLOPs / cuSPARSE GFLOPs * 100%
                pcts.append((row["mean_gflops"].values[0] / cg) * 100.0)
            else:
                pcts.append(0)
        plt.bar(x + i * width, pcts, width, label=impl)

    plt.axhline(100.0, color="green", linestyle="--", label="100% cuSPARSE Performance")
    plt.xlabel("Matrix Dataset")
    plt.ylabel("Relative Performance (% of cuSPARSE)")
    plt.title("Custom Kernel Performance Relative to cuSPARSE Reference")
    plt.xticks(x + width * (len(custom_impls) - 1) / 2, datasets, rotation=45, ha="right")
    plt.grid(True, axis="y")
    plt.legend()
    plt.savefig(PLOTS_DIR / "11_cusparse_percentage.png", dpi=300)
    plt.close()

def plot_12_ell_padding_overhead(meta_df):
    if meta_df is None or meta_df.empty:
        return
    plt.figure(figsize=(11, 6))
    sorted_meta = meta_df.sort_values(by="ell_padding_ratio")
    # Padding overhead % = (padding_ratio - 1) * 100
    overhead_pct = (sorted_meta["ell_padding_ratio"] - 1.0) * 100.0

    colors = ["salmon" if p > 500 else "skyblue" for p in overhead_pct]
    plt.bar(sorted_meta["matrix_id"], overhead_pct, color=colors, edgecolor="black")
    plt.yscale("log")
    plt.xlabel("Matrix Dataset")
    plt.ylabel("ELL Padding Memory Overhead (%) [Log Scale]")
    plt.title("ELLPACK Storage Overhead: Zero Padding Due to Row Length Variation")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y")
    plt.savefig(PLOTS_DIR / "12_ell_padding_overhead.png", dpi=300)
    plt.close()

def main():
    print("Generating all 12 publication-quality plots...")
    summary_df, raw_df, meta_df = load_data()
    if summary_df is None or summary_df.empty:
        print("No benchmark summary data available. Skipping plot generation.")
        return

    plot_1_execution_time_vs_size(summary_df)
    plot_2_gflops_comparison(summary_df)
    plot_3_speedup_vs_cpu(summary_df)
    plot_4_speedup_vs_ideal(summary_df)
    plot_5_parallel_efficiency(raw_df)
    plot_6_roofline(summary_df)
    plot_7_stacked_time_breakdown(summary_df)
    plot_8_ell_width_vs_time(summary_df, meta_df)
    plot_9_hyb_width_vs_time(raw_df)
    plot_10_achieved_bandwidth(summary_df)
    plot_11_cusparse_percentage(summary_df)
    plot_12_ell_padding_overhead(meta_df)

    print(f"All plots successfully generated in: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
