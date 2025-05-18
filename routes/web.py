"""Flask application factory for the web application."""

from pathlib import Path
from PIL import Image
from flask import Blueprint, render_template, request, current_app
from services.classifier import ImageClassifier
from services.explainers.integrated_gradients import IntegratedGradientsExplainer
from services.explainers.gradcam import GradCAMExplainer
from services.explainers.lime import LIMEExplainer
from services.explainers.shap import SHAPExplainer


web_bp = Blueprint("web", __name__)

classifier = ImageClassifier()
explainers = {
    "gradcam": GradCAMExplainer(classifier.model),
    "integrated_gradients": IntegratedGradientsExplainer(classifier.model),
    # "lime": LIMEExplainer(classifier.model),
    # "shap": SHAPExplainer(classifier.model),
}


def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return Path(filename).suffix.lower() in current_app.config["ALLOWED_EXTENSIONS"]


def validate_upload_request(files: dict) -> str:
    """Validate the image upload request."""
    if "image" not in files:
        return "No image provided"
    file = files["image"]
    if file.filename == "":
        return "No image selected"
    if not allowed_file(file.filename):
        return "File type not allowed"
    return None


@web_bp.route("/")
def home():
    """Render the home page with the upload form."""
    return render_template("index.html")


@web_bp.route("/result", methods=["POST"])
def result():
    """Process image upload and display classification results."""
    error = validate_upload_request(request.files)
    if error:
        return render_template("index.html", error=error)
    file = request.files["image"]

    try:
        image = Image.open(file.stream).convert("RGB")

        # Save original image
        base_path = current_app.config["UPLOAD_FOLDER"]
        image_path = f"{base_path}/temp_image.jpg"
        image.save(image_path)

        # Classify image
        class_id, class_name, confidence = classifier.classify(image)

        # Generate explanations using all explainers
        images = []
        for explainer_name, explainer in explainers.items():
            print(f"Explainer: {explainer_name}")
            attr_image = explainer.explain(image, class_id)
            attr_image_path = f"{base_path}/{explainer_name}_image.png"
            attr_image.save(attr_image_path)
            images.append(
                {
                    "title": explainer_name,
                    "path": str(attr_image_path),
                }
            )

        return render_template(
            "result.html",
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            original_image_path=str(image_path),
            images=images,
        )
    except (OSError, ValueError) as e:
        current_app.logger.error(f"Error processing image: {str(e)}")
        return render_template("index.html", error=f"Processing error: {str(e)}")
