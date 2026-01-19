import torch
import numpy as np
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image


def generate_gradcam_pp(model, image_tensor, target_layer):
    """
    Generate Grad-CAM++ heatmap for an image model.
    """
    cam = GradCAMPlusPlus(
        model=model,
        target_layers=[target_layer],
        use_cuda=torch.cuda.is_available()
    )

    grayscale_cam = cam(input_tensor=image_tensor)
    return grayscale_cam


def overlay_heatmap(image_rgb, cam_mask):
    """
    Overlay Grad-CAM++ heatmap on original image.
    """
    cam_mask = cam_mask[0]  # first image
    visualization = show_cam_on_image(
        image_rgb.astype(np.float32),
        cam_mask,
        use_rgb=True
    )
    return visualization
