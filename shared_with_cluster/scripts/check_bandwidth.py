from collections import Counter

path = "datasets/original/qpband.mtx"
offsets = Counter()

with open(path) as f:
    for line in f:
        line = line.strip()

        if not line or line.startswith("%"):
            continue

        parts = line.split()

        if len(parts) < 3:
            continue

        row = int(parts[0])
        col = int(parts[1])

        offsets[abs(row - col)] += 1

print("Offset ranges:")

ranges = [
    (0, 10),
    (11, 50),
    (51, 100),
    (101, 500),
    (501, 1000),
    (1001, 5000),
    (5001, 10000),
    (10001, 20000),
]

for low, high in ranges:
    count = sum(
        value for offset, value in offsets.items()
        if low <= offset <= high
    )
    print(f"{low:5}-{high:5}: {count}")

print("Distinct offsets:", len(offsets))
