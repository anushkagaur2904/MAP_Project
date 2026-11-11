import subprocess
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
bin_dir = ROOT / "bin"
bin_dir.mkdir(exist_ok=True)

exe = bin_dir / "first_touch.exe"
csv_path = ROOT / "results" / "raw" / "first_touch.csv"
csv_path.parent.mkdir(parents=True, exist_ok=True)

cmd = [
    "gcc", "-O3", "-std=c11", "-Wall", "-Wextra",
    "-fopenmp", "-march=native",
    str(ROOT / "src" / "first_touch.c"),
    "-o", str(exe)
]
subprocess.run(cmd, cwd=ROOT, check=True)

result = subprocess.run(
    [str(exe)],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True
)

print(result.stdout)

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for line in result.stdout.strip().splitlines():
        writer.writerow(line.split(","))

print("Saved first-touch measurements:", csv_path)