from pathlib import Path
import pandas as pd, numpy as np, matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
raw=ROOT/"results/raw/team5_suitesparse_raw.csv"
out=ROOT/"results/processed";plots=ROOT/"results/plots"
out.mkdir(parents=True,exist_ok=True);plots.mkdir(parents=True,exist_ok=True)
if not raw.exists(): raise SystemExit("Run benchmarks first.")
d=pd.read_csv(raw)
d=d[d["warmup"]==0].copy()
d["throughput"]=pd.to_numeric(d["throughput"],errors="coerce")
d["total_ms"]=pd.to_numeric(d["total_ms"],errors="coerce")
group=["dataset","implementation","threads_or_ranks","schedule","chunk","proc_bind"]
summary=d.groupby(group,dropna=False).agg(mean_ms=("kernel_ms","mean"),std_ms=("kernel_ms","std"),mean_gflops=("throughput","mean"),std_gflops=("throughput","std"),runs=("run_id","count"),max_error=("error","max")).reset_index()
summary.to_csv(out/"team5_summary.csv",index=False)
# 1 GFLOP/s per format per matrix (best thread/schedule is explicitly labelled)
best=summary.sort_values("mean_gflops",ascending=False).groupby(["dataset","implementation"]).head(1)
for ds,g in best.groupby("dataset"):
    plt.figure(figsize=(8,4));plt.bar(g["implementation"],g["mean_gflops"],yerr=g["std_gflops"].fillna(0),capsize=4)
    plt.ylabel("GFLOP/s");plt.title(f"Best measured GFLOP/s – {ds}");plt.tight_layout();plt.savefig(plots/f"gflops_{ds}.png",dpi=180);plt.close()
# 2 speedup vs threads with propagated standard-deviation error bars
seq_rows=summary[summary.implementation.str.endswith("_seq")].copy()
seq_lookup={}
for _,r in seq_rows.iterrows():
    fmt=str(r.implementation).split("_")[0]
    seq_lookup[(r.dataset,fmt,str(r.proc_bind))]=(r.mean_ms,0.0 if pd.isna(r.std_ms) else r.std_ms)
rows=[]
for _,r in summary.iterrows():
    fmt=str(r.implementation).split("_")[0]
    if (r.dataset,fmt,str(r.proc_bind)) not in seq_lookup: continue
    base,base_sd=seq_lookup[(r.dataset,fmt,str(r.proc_bind))]
    par_sd=0.0 if pd.isna(r.std_ms) else r.std_ms
    speed=base/r.mean_ms
    speed_sd=speed*np.sqrt((base_sd/base)**2+(par_sd/r.mean_ms)**2) if base and r.mean_ms else 0
    rows.append((r.dataset,fmt,r.threads_or_ranks,speed,speed_sd,r.schedule,r.proc_bind))
sp=pd.DataFrame(rows,columns=["dataset","format","threads","speedup","speedup_sd","schedule","proc_bind"])
for ds,g in sp.groupby("dataset"):
    plt.figure(figsize=(7,4))
    for (fmt,pb),h in g.groupby(["format","proc_bind"]):
        z=h.groupby("threads").agg(speed=("speedup","mean"),sd=("speedup_sd","mean"))
        plt.errorbar(z.index,z.speed,yerr=z.sd,marker="o",capsize=3,label=f"{fmt}, bind={pb}")
    mx=max(g.threads);plt.plot([1,mx],[1,mx],"--",label="ideal");plt.xlabel("Threads");plt.ylabel("Speedup");plt.title(f"Speedup – {ds}");plt.legend();plt.tight_layout();plt.savefig(plots/f"speedup_{ds}.png",dpi=180);plt.close()
# 3 schedule comparison
omp=d[d.implementation.str.endswith("_omp")].copy()
if not omp.empty:
    z=omp.groupby("schedule").throughput.agg(["mean","std"])
    plt.figure(figsize=(6,4));plt.bar(z.index,z["mean"],yerr=z["std"].fillna(0),capsize=4);plt.ylabel("Mean GFLOP/s");plt.title("OpenMP scheduling comparison");plt.tight_layout();plt.savefig(plots/"scheduling_comparison.png",dpi=180);plt.close()
# 4 static row-partition load imbalance vs matrix irregularity
imb=ROOT/"datasets/metadata/static_partition_imbalance.csv"
if imb.exists():
    q=pd.read_csv(imb)
    plt.figure(figsize=(7,4))
    for nt,h in q.groupby("threads"):
        plt.errorbar(h.irregularity,h.static_partition_imbalance,yerr=np.zeros(len(h)),fmt="o",capsize=2,label=f"{nt} threads (deterministic partition metric)")
    plt.xlabel("Matrix irregularity (max NNZ/row ÷ mean NNZ/row)");plt.ylabel("Static partition imbalance (max thread NNZ ÷ mean thread NNZ)")
    plt.title("Static row-partition load imbalance vs matrix irregularity");plt.legend();plt.tight_layout();plt.savefig(plots/"load_imbalance_vs_irregularity.png",dpi=180);plt.close()
# 5 Bandwidth roofline: measured STREAM bandwidth; compute ceiling unavailable.
rp=ROOT/"config/roofline.json"
meta=ROOT/"datasets/metadata/matrix_metadata.csv"
if rp.exists() and meta.exists():
    import json
    rc=json.loads(rp.read_text())
    bw=float(rc.get("stream_bandwidth_gbs",0))

    if bw>0:
        x=np.logspace(-3,3,300)
        y=bw*x

        plt.figure(figsize=(7,5))
        plt.loglog(x,y,label=f"STREAM bandwidth roof ({bw:.4f} GB/s)")

        md=pd.read_csv(meta)
        pts=summary.merge(md,left_on="dataset",right_on="matrix_id")

        for fmt,h in pts.groupby(pts.implementation.str.split("_").str[0]):
            if fmt=="coo":
                bytes_nz=np.full(len(h),24.0)
            elif fmt=="csr":
                bytes_nz=20.0+4.0*h.rows.to_numpy()/h.nnz.to_numpy()
            else:
                bytes_nz=8.0+12.0*h.ell_padding_ratio.to_numpy()

            ai=2.0/bytes_nz
            plt.errorbar(ai,h.mean_gflops.to_numpy(),
                         yerr=h.std_gflops.fillna(0).to_numpy(),
                         fmt="o",capsize=3,label=f"{fmt} measured")

        plt.xlabel("Arithmetic intensity (FLOP/byte)")
        plt.ylabel("GFLOP/s")
        plt.title("SpMV bandwidth roofline (compute ceiling unavailable)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots/"roofline.png",dpi=180)
        plt.close()
print("Processed results:",out/"team5_summary.csv")
print("Plots:",plots)
