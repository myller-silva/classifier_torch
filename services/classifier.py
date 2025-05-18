"""Classifier service for pre-trained torchvision models."""

from typing import Tuple
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import quantization
from PIL import Image


def create_model(
    model_name: str, quantize: bool = False
) -> Tuple[torch.nn.Module, list]:
    """
    Create a pre-trained model with specified name and quantization.

    Args:
        model_name: Model name (e.g., 'resnet18', 'resnet50').
        quantize: Use quantized model if available (default: False).

    Returns:
        Tuple of (model instance, list of class names).

    Raises:
        ValueError: If model or quantized version is not supported.
    """
    model_config = {
        "resnet18": {
            "standard": (models.resnet18, models.ResNet18_Weights.DEFAULT),
            "quantized": (
                quantization.resnet18,
                quantization.ResNet18_QuantizedWeights.DEFAULT,
            ),
        },
        "resnet50": {
            "standard": (models.resnet50, models.ResNet50_Weights.DEFAULT),
            "quantized": (
                quantization.resnet50,
                quantization.ResNet50_QuantizedWeights.DEFAULT,
            ),
        },
    }

    if model_name not in model_config:
        raise ValueError(f"Supported models: {list(model_config.keys())}")

    config = model_config[model_name]
    model_type = "quantized" if quantize and "quantized" in config else "standard"

    if quantize and model_type != "quantized":
        raise ValueError(f"Quantized {model_name} not available")

    model_fn, weights = config[model_type]
    kwargs = {"weights": weights}
    if model_type == "quantized":
        kwargs["quantize"] = True

    model = model_fn(**kwargs)
    model.eval()
    classes = weights.meta["categories"]
    return model, classes


class ImageClassifier:
    """Service for classifying images with a pre-trained model."""

    def __init__(
        self,
        model: torch.nn.Module = None,
        classes: list = None,
    ):
        """
        Initialize classifier with a pre-trained model and class names.

        Args:
            model: Pre-trained PyTorch model in eval mode.
            classes: List of class names (e.g., ImageNet categories).
        """
        if model is None or classes is None:
            model, classes = create_model("resnet50", quantize=False) # Default model
        self.model = model
        self.classes = classes
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def classify(self, image: Image.Image) -> Tuple[int, str, float]:
        """
        Classify an image.

        Args:
            image: PIL Image in RGB format.

        Returns:
            Tuple of (class ID, class name, confidence score).
        """
        image = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(image)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        class_id = predicted.item()
        return class_id, self.classes[class_id], confidence.item()
