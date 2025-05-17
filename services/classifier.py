"""Classifier service using a pre-trained ResNet18 model."""

from typing import Tuple
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image


class ImageClassifier:
    """Service for classifying images using a pre-trained ResNet18 model."""

    def __init__(self):
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.classes = ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def classify(self, image: Image.Image) -> Tuple[int, str, float]:
        """
        Classify an image and return the class ID, name, and confidence.

        Args:
            image: PIL Image object in RGB format

        Returns:
            Tuple containing class ID, class name, and confidence score
        """
        image = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(image)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        class_id = predicted.item()
        class_name = self.classes[class_id]
        return class_id, class_name, confidence.item()
