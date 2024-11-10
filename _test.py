import lucid
import lucid.nn as nn

# Example: Linear layer with input size 2, output size 3
linear_layer = nn.Linear(in_features=2, out_features=4)

# Input tensor (batch size 3, input size 2)
input_tensor = lucid.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)

# Perform forward pass
output = linear_layer(input_tensor)
print(output)
# Perform backward pass
output.sum().backward()  # Sum of the output as a simple loss

# Print the gradient of the bias
print("Gradient of bias:", linear_layer.bias_.grad)
