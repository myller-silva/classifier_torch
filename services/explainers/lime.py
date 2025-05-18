"""Explainer service for generating explanations using LIME."""

from typing import Optional
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries


class LIMEExplainer:
    """Service for generating explanations using LIME."""

    def __init__(
        self,
        model: torch.nn.Module,
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
        Initialize LIME explainer.

        Args:
            model: Pre-trained PyTorch model in eval mode.
            transform: Image transformation pipeline (default: standard ImageNet transform).
        """
        self.model = model.eval()
        self.transform = transform
        self.device = next(model.parameters()).device

    def _predict_fn(self, images: np.ndarray) -> np.ndarray:
        """Helper function to predict probabilities for LIME."""
        batch = torch.stack(
            [self.transform(Image.fromarray(img.astype(np.uint8))) for img in images]
        ).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(batch), dim=1)
        return probs.cpu().numpy()

    def explain(self, image: Image.Image, target: int) -> Image.Image:
        """
        Generate a LIME explanation for the given image and target class.

        Args:
            image: PIL Image object in RGB format.
            target: Target class ID for explanation.

        Returns:
            PIL Image object representing the LIME attribution map.
        """
        # Convert image to numpy array
        img_array = np.array(image)

        # Create LIME explainer
        explainer = lime_image.LimeImageExplainer()

        # Generate explanation
        explanation = explainer.explain_instance(
            img_array,
            self._predict_fn,
            top_labels=1,
            num_samples=10,  # Adjust for speed vs. accuracy
            random_seed=42,
        )

        # Get positive regions for target class
        _, mask = explanation.get_image_and_mask(
            target, positive_only=True, num_features=5, hide_rest=False
        )

        # Create heatmap
        heatmap = mark_boundaries(img_array / 255.0, mask) * 255
        heatmap = heatmap.astype(np.uint8)
        return Image.fromarray(heatmap)
