import lucid

lucid.random.seed(42)


arr = [
    lucid.tensor([1, 2, 3], requires_grad=True),
    lucid.tensor([4, 5, 6], requires_grad=True),
    lucid.tensor([7, 8, 9], requires_grad=True),
]

B = lucid.stack(*arr, axis=0) ** 2 / 2
B.backward()


print(*arr, sep="\n")
print(*[a.grad for a in arr], sep="\n")
