import torch

print(torch.__version__)

for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(i).name)