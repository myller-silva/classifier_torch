"""Explainer service for generating explanations using Integrated Gradients."""
from typing import Optional
from PIL import Image
import numpy as np
from captum.attr import IntegratedGradients
import torch
from torchvision import transforms


class ImageExplainer:
    """Service for generating explanations using Integrated Gradients."""

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
        self.model = model
        self.transform = transform
        self.ig = IntegratedGradients(self.model)

    def explain(self, image: Image.Image, target: int) -> Image.Image:
        """
        Generate an explanation for the given image and target class.

        Args:
            image: PIL Image object in RGB format
            target: Target class ID for explanation

        Returns:
            PIL Image object representing the attribution map
        """
        image = self.transform(image).unsqueeze(0)
        attributions = self.ig.attribute(image, target=target)
        attr_image = attributions.squeeze().detach().numpy()
        attr_image = np.transpose(attr_image, (1, 2, 0))
        attr_image = (attr_image - attr_image.min()) / (
            attr_image.max() - attr_image.min() + 1e-8
        )
        attr_image = Image.fromarray((attr_image * 255).astype(np.uint8))
        return attr_image
