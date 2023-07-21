import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from model import Net


def print_model_summary():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net().to(device)
    summary(model, input_size=(1, 28, 28))