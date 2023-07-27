# check pytorch version
import torch
# from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
data = [[1, 2], [3, 4]]
# torch.device("mps")
x_data = torch.tensor(data)
print(x_data)
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print("Random Tensor:", rand_tensor)
print(f"Ones Tensor:", ones_tensor)
print(f"Zeros Tensor;", zeros_tensor)
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())