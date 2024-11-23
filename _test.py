import lucid

arr = [
    lucid.tensor([[1, 2]], requires_grad=True),
    lucid.tensor([[3, 4]], requires_grad=True),
]

B = lucid.vstack(arr)
B.backward()

print(B)
print(*[a.grad for a in arr], sep="\n")
