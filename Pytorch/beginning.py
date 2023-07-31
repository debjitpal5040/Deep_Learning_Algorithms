import torch
from torch import tensor, rand

tensor = torch.rand(3, 4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")

a = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([6.0], requires_grad=True)
Q = 3 * a**3 - b**2
Q.backward()
print(f"a.grad: {a.grad}")
print(f"b.grad: {b.grad}")
print(f"a: {a}")
print(f"b: {b}")
print(f"Q: {Q}")
