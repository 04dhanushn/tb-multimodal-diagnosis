import torch
import numpy as np

def run_fusion_inference(model, tabular_tensor, image_tensor, device):
    """
    Run inference using fusion model
    """
    model.eval()
    with torch.no_grad():
        tabular_tensor = tabular_tensor.to(device)
        image_tensor = image_tensor.to(device)

        logits = model(tabular_tensor, image_tensor)
        prob = torch.sigmoid(logits)

    return prob.cpu().numpy()
