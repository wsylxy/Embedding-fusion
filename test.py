file = open(file="ckpt/data/test.txt", mode="r", encoding="utf-8")
for i, line in enumerate(file):
    expr = line.strip().split(sep='\t')
    print(expr)