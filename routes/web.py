"""Flask application factory for the web application."""

# import os
# from werkzeug.utils import secure_filename
from PIL import Image
from pathlib import Path
from flask import Blueprint, render_template, request, current_app
from services.classifier import ImageClassifier
from services.explainer import ImageExplainer

web_bp = Blueprint("web", __name__)
classifier = ImageClassifier()
explainer = ImageExplainer(classifier.model)

def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return Path(filename).suffix.lower() in current_app.config["ALLOWED_EXTENSIONS"]

@web_bp.route("/")
def home():
    """Render the home page with the upload form."""
    return render_template("index.html")

@web_bp.route("/result", methods=["POST"])
def result():
    """Process image upload and display classification results."""
    if "image" not in request.files:
        return render_template("index.html", error="No image provided")
    
    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", error="No image selected")
    
    if not allowed_file(file.filename):
        return render_template("index.html", error="Invalid file type. Allowed: jpg, jpeg, png")

    try:
        image = Image.open(file.stream).convert("RGB")
        
        # Save original image
        image_path = Path(current_app.config["UPLOAD_FOLDER"]) / "temp_image.jpg"
        image.save(image_path)

        # Classify image
        class_id, class_name, confidence = classifier.classify(image)

        # Generate explanation
        attr_image = explainer.explain(image, class_id)
        attr_image_path = Path(current_app.config["UPLOAD_FOLDER"]) / "attr_image.png"
        attr_image.save(attr_image_path)

        return render_template(
            "result.html",
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            image_path="uploads/temp_image.jpg",
            attr_image_path="uploads/attr_image.png"
        )
    except Exception as e:
        current_app.logger.error(f"Error processing image: {str(e)}")
        return render_template("index.html", error=f"Processing error: {str(e)}")