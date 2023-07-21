import torch.nn as nn
from model import Net

def calculate_receptive_field(model):
    receptive_field = 1
    stride = 1
    padding = 0
    kernel_size = 1

    for layer in model.children():
        if isinstance(layer, nn.Conv2d):
            dilation = layer.dilation[0]
            kernel_size = layer.kernel_size[0]
            stride = layer.stride[0]
            padding = layer.padding[0]
            receptive_field = ((receptive_field - 1) * stride) + (dilation * (kernel_size - 1)) + 1

        if isinstance(layer, nn.ReLU):
            receptive_field = ((receptive_field - 1) * stride) + 1

    return receptive_field, kernel_size, stride, padding

# Create an instance of the DilatedConvNet model
model = Net()

# Calculate the receptive field for the model
receptive_field, kernel_size, stride, padding = calculate_receptive_field(model)

print(f"Receptive Field: {receptive_field}")
print(f"Kernel Size: {kernel_size}")
print(f"Stride: {stride}")
print(f"Padding: {padding}")
