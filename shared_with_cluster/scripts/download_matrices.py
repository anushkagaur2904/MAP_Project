"""Download SuiteSparse Matrix Collection entries listed in config/matrices.csv.
Expected columns: matrix_id,relative_path,category,group,name.
The standard SuiteSparse archive URL pattern is https://sparse.tamu.edu/MM/<group>/<name>.tar.gz.
Check each entry on https://sparse.tamu.edu/ before running; archive names can change.
"""
import csv,tarfile,urllib.request,io
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader(open(ROOT/"config/matrices.csv",newline="",encoding="utf-8-sig")))
for r in rows:
    dest=ROOT/r["relative_path"]
    if dest.exists(): print("EXISTS",r["matrix_id"]);continue
    group=r.get("group","").strip();name=r.get("name","").strip()
    if not group or not name:
        print("NO DOWNLOAD METADATA",r["matrix_id"],"-- add group and name after checking SuiteSparse page");continue
    url=f"https://sparse.tamu.edu/MM/{group}/{name}.tar.gz"
    print("Downloading",url)
    try:
        data=urllib.request.urlopen(url,timeout=120).read()
        with tarfile.open(fileobj=io.BytesIO(data),mode="r:gz") as tf:
            members=[m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".mtx")]
            if not members: raise RuntimeError("archive contains no .mtx file")
            member=members[0];src=tf.extractfile(member)
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(src.read())
        print("SAVED",dest)
    except Exception as e: print("FAILED",r["matrix_id"],e)
