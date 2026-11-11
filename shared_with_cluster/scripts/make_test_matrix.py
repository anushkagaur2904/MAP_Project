from pathlib import Path
p=Path("datasets/original/test_small.mtx");p.parent.mkdir(parents=True,exist_ok=True)
p.write_text("""%%MatrixMarket matrix coordinate real general
% tiny deterministic correctness test
4 4 6
1 1 2
1 4 3
2 2 4
3 1 5
3 3 6
4 4 7
""",encoding="ascii")
Path("datasets/vectors").mkdir(parents=True,exist_ok=True)
Path("datasets/vectors/test_small_x.txt").write_text("1\n2\n3\n4\n",encoding="ascii")
print("Created test_small.mtx and fixed x vector. Expected y: 14, 8, 23, 28.")
