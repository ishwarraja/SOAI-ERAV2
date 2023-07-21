import torch
from torchsummary import summary

def check_parameters(model):
    # Step 5: Total Params must be less than 200k
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    # Step 6: Get Torch Summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, input_size=(32, 32, 32))
