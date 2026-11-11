import csv, hashlib, pathlib, random, sys
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
cfg=ROOT/"config/matrices.csv"
vectors=ROOT/"datasets/vectors"; meta=ROOT/"datasets/metadata"
vectors.mkdir(parents=True,exist_ok=True);meta.mkdir(parents=True,exist_ok=True)

def read_mtx(path):
    with open(path,encoding="utf-8",errors="replace") as f:
        banner=f.readline().strip().lower()
        if "matrixmarket matrix coordinate" not in banner: raise ValueError(f"{path}: expected coordinate Matrix Market")
        field=banner.split()[3]; sym=banner.split()[4]
        line=f.readline()
        while line.startswith("%"): line=f.readline()
        m,n,nz=map(int,line.split()[:3])
        rows=[];cols=[];vals=[]
        for line in f:
            if not line.strip() or line.startswith("%"): continue
            p=line.split(); r=int(p[0])-1;c=int(p[1])-1
            v=1.0 if field=="pattern" else float(p[2])
            rows.append(r);cols.append(c);vals.append(v)
            if sym in ("symmetric","hermitian") and r!=c:
                rows.append(c);cols.append(r);vals.append(v)
    return m,n,np.asarray(rows,dtype=np.int64),np.asarray(cols,dtype=np.int64),np.asarray(vals,dtype=np.float64)

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

records=[]
partition_rows=[]
with open(cfg,newline="",encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        if not row.get("matrix_id"):continue
        p=ROOT/row["relative_path"]
        if not p.exists():
            print(f"SKIP missing: {p}");continue
        m,n,r,c,v=read_mtx(p)
        counts=np.bincount(r,minlength=m)
        rng=np.random.default_rng(int(hashlib.sha256(row["matrix_id"].encode()).hexdigest()[:8],16))
        x=rng.random(n,dtype=np.float64)
        xp=vectors/f'{row["matrix_id"]}_x.txt'
        np.savetxt(xp,x,fmt="%.17g")
        records.append({
          "matrix_id":row["matrix_id"],"file":str(p.relative_to(ROOT)),"category":row["category"],
          "rows":m,"cols":n,"nnz":len(v),"mean_nnz_per_row":float(counts.mean()),
          "std_nnz_per_row":float(counts.std()),"max_nnz_per_row":int(counts.max(initial=0)),
          "irregularity":float(counts.max(initial=0)/counts.mean()) if counts.mean() else 0,
          "ell_width":int(counts.max(initial=0)),
          "ell_padding_ratio":float(m*max(1,int(counts.max(initial=0)))/max(1,len(v))),
          "matrix_sha256":sha(p),"vector_file":str(xp.relative_to(ROOT)),
          "vector_sha256":sha(xp)
        })
        for nt in (1,2,4,8,16):
            # OpenMP static contiguous row partition: estimate work as NNZ assigned per thread.
            work=[]
            for t in range(nt):
                lo=(m*t)//nt; hi=(m*(t+1))//nt
                work.append(int(counts[lo:hi].sum()))
            avg=(sum(work)/nt) if nt else 0
            partition_rows.append({"matrix_id":row["matrix_id"],"threads":nt,
              "irregularity":float(counts.max(initial=0)/counts.mean()) if counts.mean() else 0,
              "max_thread_nnz":max(work,default=0),"mean_thread_nnz":avg,
              "static_partition_imbalance":(max(work)/avg) if avg else 0})
with open(meta/"matrix_metadata.csv","w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]) if records else ["matrix_id"]);w.writeheader();w.writerows(records)
with open(meta/"static_partition_imbalance.csv","w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(partition_rows[0]) if partition_rows else ["matrix_id"]);w.writeheader();w.writerows(partition_rows)
print(f"Prepared {len(records)} matrices. Metadata: {meta/'matrix_metadata.csv'}")
if len(records)<10: print("WARNING: assignment asks for 10–15 matrices; add more entries to config/matrices.csv.")
