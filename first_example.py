from __future__ import print_function
import torch
import numpy as np

x = torch.Tensor(5, 3)
print(x)
x = torch.rand(5, 3)
print(x)
y = torch.rand(5, 3)
print(x + y)
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
