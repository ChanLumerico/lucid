from lucid.datasets import MNIST

mnist_train = MNIST(root="./data", train=True, download=True)

image, label = mnist_train[0]

print(f"Image Shape: {image.shape}, Label: {label}")
