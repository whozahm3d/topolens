"""Grad-CAM implementation for Topolens."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm

class GradCAM:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self.target_layer = self.model.features[3]

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor: torch.Tensor, target_index: int) -> np.ndarray:
        """
        Generate Grad-CAM for a given input tensor and target index (0 for vertices, 1 for edges).
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass
        target = output[0, target_index]
        target.backward(retain_graph=True)
        
        # Compute CAM
        # gradients shape: [batch, channels, H, W]
        # activations shape: [batch, channels, H, W]
        gradients = self.gradients.detach()
        activations = self.activations.detach()
        
        # Global average pool the gradients over spatial dimensions (H, W)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted sum of activations over channels
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU
        cam = F.relu(cam)
        
        # Upsample to 224x224
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Min-max normalization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam

def overlay_cam(img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Overlay a computed CAM onto a PIL image."""
    img = img.resize((224, 224))
    
    # Apply colormap to CAM
    colormap = matplotlib.colormaps['jet']
    cam_colored = colormap(cam) # RGBA [224, 224, 4]
    
    # Convert RGBA [0,1] to RGB [0,255] PIL image
    cam_img = Image.fromarray(np.uint8(cam_colored[:, :, :3] * 255))
    
    # Alpha blend
    return Image.blend(img.convert('RGB'), cam_img, alpha)
