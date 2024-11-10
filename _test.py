import lucid
import numpy as np


x = lucid.tensor(np.random.randn(1), requires_grad=True)
y = lucid.tensor(np.random.randn(1), requires_grad=True)

learning_rate = 0.01
num_iterations = 100

for iteration in range(num_iterations):
    f = x**2 - y**2
    f.backward()

    x.data -= learning_rate * x.grad
    y.data -= learning_rate * y.grad

    x.grad = None
    y.grad = None

    if iteration % 10 == 0:
        print(f)