import torch

a = torch._C._cuda_getDeviceCount()
print(a)
