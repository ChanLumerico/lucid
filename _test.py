import lucid
import lucid.nn as nn

lucid.random.seed(42)


lin_1 = nn.Linear(in_features=2, out_features=4)
lin_2 = nn.Linear(in_features=4, out_features=6)

x = lucid.random.randn((3, 2), requires_grad=True)
y1 = lin_1(x)
y2 = lin_2(y1)

y2.backward()

print(f"y2\n{y2}")
print(f"dy1:\n{y1.grad}")
print(f"dx:\n{x.grad}")
