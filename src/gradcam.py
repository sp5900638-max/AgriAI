"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for CropSense AI
Generates visual explanations highlighting leaf regions driving CNN predictions.
Integrates directly with PyTorch models and OpenCV visualization overlays.
"""

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class GradCAM:
    """
    Grad-CAM implementation for PyTorch convolutional backbones.
    Hooks into activations and gradients of the target convolutional layer.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Computes Grad-CAM heatmap for a given input tensor and target class.
        input_tensor: shape (1, C, H, W)
        Returns:
            heatmap: 2D numpy array (H, W) in [0.0, 1.0]
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for the target score
        target_score = output[0, target_class]
        target_score.backward()

        # Global average pooling of gradients
        # Gradients: (1, K, h, w) -> weights: (K,)
        weights = torch.mean(self.gradients[0], dim=(1, 2))

        # Weighted combination of activation maps
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32, device=input_tensor.device)
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i]

        # ReLU on combined map
        cam = F.relu(cam)
        cam = cam.cpu().numpy()

        # Normalize to [0, 1]
        if np.max(cam) > 0:
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        else:
            cam = np.zeros_like(cam)

        return cam


def overlay_gradcam(
    image_bgr: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlays Grad-CAM heatmap onto original BGR image using OpenCV.
    """
    h, w = image_bgr.shape[:2]
    # Resize heatmap to input dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Color map
    colored_map = cv2.applyColorMap(heatmap_uint8, colormap)

    # Blend
    overlay = cv2.addWeighted(colored_map, alpha, image_bgr, 1 - alpha, 0)
    return overlay
