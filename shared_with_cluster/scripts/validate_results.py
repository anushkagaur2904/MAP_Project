import csv,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
errors=[];warnings=[]
meta=ROOT/"datasets/metadata/matrix_metadata.csv"
raw=ROOT/"results/raw/team5_raw.csv"
if not meta.exists(): errors.append("matrix metadata missing; run scripts/prepare_inputs.py")
else:
    with open(meta,newline="",encoding="utf-8-sig") as f: matrices=list(csv.DictReader(f))
    if not 10 <= len(matrices) <= 15: errors.append(f"assignment requires 10–15 matrices; currently prepared {len(matrices)}")
    cats={r.get("category","").lower() for r in matrices}
    if not any("regular" in c or "band" in c for c in cats): warnings.append("No matrix marked regular/banded")
    if not any("power" in c or "irregular" in c for c in cats): warnings.append("No matrix marked power_law/irregular")
    if not any("large" in c for c in cats): warnings.append("No matrix marked large")
if not raw.exists(): errors.append("raw benchmark CSV missing; run scripts/run_benchmarks.py")
else:
    with open(raw,newline="",encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
    required="cluster problem platform implementation dataset size threads_or_ranks run_id total_ms kernel_ms transfer_ms throughput error power_w warmup schedule chunk first_touch proc_bind".split()
    if rows:
        missing=[c for c in required if c not in rows[0]]
        if missing: errors.append("CSV missing columns: "+", ".join(missing))
        groups={}
        for r in rows:
            try:
                if float(r["error"])>1e-10: errors.append(f"correctness error exceeds 1e-10 for {r['dataset']} {r['implementation']}")
                if int(r["warmup"])==0:
                    k=(r["dataset"],r["implementation"],r["threads_or_ranks"],r["schedule"],r["chunk"],r["proc_bind"])
                    groups[k]=groups.get(k,0)+1
            except Exception as e: errors.append(f"invalid row {r}: {e}")
        low=[(k,n) for k,n in groups.items() if n<5]
        if low: errors.append(f"{len(low)} configurations have fewer than 5 measured runs")
    else: errors.append("raw CSV has no measurement rows")
if errors:
    print("VALIDATION FAILED")
    for x in errors: print("ERROR:",x)
else: print("Core CSV and measurement checks passed")
for x in warnings: print("WARNING:",x)
print("This validator cannot verify STREAM values, hardware facts, plot interpretation, or whether matrices are correctly categorized. Review those manually.")
sys.exit(1 if errors else 0)
