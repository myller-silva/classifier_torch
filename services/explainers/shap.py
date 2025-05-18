"""Explainer service for generating explanations using SHAP."""

from typing import Optional
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
# import shap
from shap import DeepExplainer


class SHAPExplainer:
    """Service for generating explanations using SHAP (Deep SHAP)."""

    def __init__(
        self,
        model: torch.nn.Module,
        transform: Optional[transforms.Compose] = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        ),
    ):
        """
        Initialize SHAP explainer.

        Args:
            model: Pre-trained PyTorch model in eval mode.
            transform: Image transformation pipeline (default: standard ImageNet transform).
        """
        self.model = model.eval()
        self.transform = transform
        self.device = next(model.parameters()).device

    def explain(self, image: Image.Image, target: int) -> Image.Image:
        """
        Generate a SHAP explanation for the given image and target class.

        Args:
            image: PIL Image object in RGB format.
            target: Target class ID for explanation.

        Returns:
            PIL Image object representing the SHAP attribution map.
        """
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Create SHAP DeepExplainer
        explainer = DeepExplainer(self.model, input_tensor)

        # Compute SHAP values
        shap_values = explainer.shap_values(input_tensor)
        shap_values = shap_values[target]  # Select SHAP values for target class

        # Convert to image
        attr_image = shap_values.squeeze().transpose(1, 2, 0)
        attr_image = (attr_image - attr_image.min()) / (
            attr_image.max() - attr_image.min() + 1e-8
        )
        attr_image = (attr_image * 255).astype(np.uint8)
        return Image.fromarray(attr_image).convert("RGB")
