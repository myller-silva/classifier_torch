"""Explainer service for generating explanations using Grad-CAM."""

from typing import Optional
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np


class GradCAMExplainer:
    """Service for generating explanations using Grad-CAM."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: str = "layer4",
        transform: Optional[transforms.Compose] = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # ),
            ]
        ),
    ):
        """
        Initialize Grad-CAM explainer.

        Args:
            model: Pre-trained PyTorch model in eval mode.
            target_layer: Name of the target convolutional layer (default: 'layer4' for ResNet).
            transform: Image transformation pipeline (default: standard ImageNet transform).
        """
        self.model = model.eval()
        self.target_layer = target_layer
        self.transform = transform
        self.gradients = None
        self.activations = None

        # Register hooks to capture gradients and activations
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        # Find the target layer
        target_module = dict(self.model.named_modules()).get(self.target_layer)
        if not target_module:
            raise ValueError(f"Target layer '{self.target_layer}' not found in model")

        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)

    def explain(self, image: Image.Image, target: int) -> Image.Image:
        """
        Generate a Grad-CAM explanation for the given image and target class.

        Args:
            image: PIL Image object in RGB format.
            target: Target class ID for explanation.

        Returns:
            PIL Image object representing the Grad-CAM heatmap.
        """
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0)
        input_tensor.requires_grad = True

        # Forward pass
        output = self.model(input_tensor)
        self.model.zero_grad()

        # Backward pass for target class
        output[:, target].backward()

        # Compute Grad-CAM
        gradients = self.gradients
        activations = self.activations
        pooled_gradients = torch.mean(gradients, dim=[2, 3], keepdim=True)
        cam = (pooled_gradients * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # Apply ReLU to remove negative values
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().numpy()

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = (cam * 255).astype(np.uint8)
        heatmap = Image.fromarray(cam).convert("RGB")

        return heatmap
