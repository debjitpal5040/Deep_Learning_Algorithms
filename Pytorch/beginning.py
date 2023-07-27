import torch
import numpy as np
# tensor = torch.rand(3, 4)
# print(f"Shape of tensor: {tensor.shape}")
# print(f"Datatype of tensor: {tensor.dtype}")
# print(f"Device tensor is stored on: {tensor.device}")
# # We move our tensor to the GPU if available
# if torch.cuda.is_available():
#     tensor = tensor.to('cuda')
#     print(f"Device tensor is stored on: {tensor.device}")
# t = torch.ones(5)
# print(f"t: {t}")
# n = t.numpy()
# print(f"n: {n}")
# t.add_(1)
# print(f"t: {t}")
# print(f"n: {n}")
# # We can also use torch.from_numpy() to convert numpy arrays to tensors
# n = np.ones(5)
# t = torch.from_numpy(n)
# print(f"t: {t}")
# print(f"n: {n}")

a = torch.tensor([2.], requires_grad=True)
b = torch.tensor([6.], requires_grad=True)
Q = 3*a**3 - b**2
Q.backward()
print(f"a.grad: {a.grad}")
print(f"b.grad: {b.grad}")
print(f"a: {a}")
print(f"b: {b}")
print(f"Q: {Q}")