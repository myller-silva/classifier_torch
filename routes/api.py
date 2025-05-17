"""Classifier service using a pre-trained ResNet18 model."""

# from werkzeug.utils import secure_filename
import io
import base64
from pathlib import Path
from PIL import Image
from flask import Blueprint, request, jsonify, current_app
from services.classifier import ImageClassifier
from services.explainer import ImageExplainer

api_bp = Blueprint("api", __name__)
classifier = ImageClassifier()
explainer = ImageExplainer(classifier.model)


def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return Path(filename).suffix.lower() in current_app.config["ALLOWED_EXTENSIONS"]


@api_bp.route("/classify", methods=["POST"])
def classify():
    """Classify an uploaded image and return class details."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Allowed: jpg, jpeg, png"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        class_id, class_name, confidence = classifier.classify(image)
        return jsonify(
            {"class_id": class_id, "class_name": class_name, "confidence": confidence}
        )
    except (IOError, ValueError) as e:
        current_app.logger.error(f"Error classifying image: {str(e)}")
        return jsonify({"error": f"Processing error: {str(e)}"}), 500


@api_bp.route("/explain", methods=["POST"])
def explain():
    """Generate explanation for an uploaded image and class ID."""
    if "image" not in request.files or "class_id" not in request.form:
        return jsonify({"error": "Missing image or class_id"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Allowed: jpg, jpeg, png"}), 400

    try:
        class_id = int(request.form["class_id"])
        image = Image.open(file.stream).convert("RGB")
        attr_image = explainer.explain(image, class_id)

        buffered = io.BytesIO()
        attr_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({"explanation": img_str})
    except ValueError:
        return jsonify({"error": "Invalid class_id format"}), 400
    except (IOError, OSError) as e:
        current_app.logger.error(f"Error generating explanation: {str(e)}")
        return jsonify({"error": f"Processing error: {str(e)}"}), 500
