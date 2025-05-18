# Image Classification and XAI API

This project provides a Flask-based web application and API for image classification using a pre-trained ResNet18 model and explainability through Integrated Gradients (XAI). It includes a user-friendly web interface for uploading images and viewing classification results, along with an API for programmatic access.

## Features

- **Image Classification:** Classify images using ResNet18 pre-trained on ImageNet, returning class ID, name, and confidence.
- **Explainability:** Generate visual explanations of classifications using Integrated Gradients.
- **Web Interface:** Upload images and view results with original and explanation images.
- **REST API:** Endpoints for classification (`/api/classify`) and explanation (`/api/explain`).
- **Docker Support:** Containerized application for easy deployment.

## Project Structure

```
project/
├── app/
│   ├── __init__.py          # Flask app initialization
│   ├── config.py            # Configuration settings
│   ├── services/
│   │   ├── classifier.py    # Image classification logic
│   │   ├── explainer.py     # XAI explanation logic
│   ├── routes/
│   │   ├── web.py           # Web interface routes
│   │   ├── api.py           # API routes
│   ├── templates/
│   │   ├── index.html       # Upload form
│   │   ├── result.html      # Classification results
│   ├── static/
│   │   ├── favicon.ico      # Favicon (optional)
│   │   ├── uploads/         # Temporary image storage
├── Dockerfile               # Docker configuration
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
```

## Prerequisites

- Python 3.8+
- Docker (optional, for containerized deployment)
- NVIDIA GPU and CUDA (optional, for GPU acceleration)

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd project
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Add a favicon:**
   Place a `favicon.ico` file in `app/static/` to avoid 404 errors, or remove the favicon route in `app/routes/web.py`.

## Running Locally

1. **Set the Flask app environment variable:**

   ```bash
   export FLASK_APP=app
   ```

   On Windows:

   ```powershell
   $env:FLASK_APP = "app"
   ```

2. **Run the application:**

   ```bash
   flask run
   ```

   The app will be available at `http://127.0.0.1:5000`.

## Running with Docker

1. **Build the Docker image:**

   ```bash
   docker build -t image-classifier .
   ```

2. **Run the container:**

   ```bash
   docker run -p 5000:5000 image-classifier
   ```

   The app will be available at `http://localhost:5000`.

## Usage

### Web Interface

1. Open `http://localhost:5000` in a browser.
2. Upload an image (JPEG or PNG).
3. View the results, including:
   - Predicted class name and ID
   - Confidence score (percentage)
   - Original image
   - XAI explanation image

### API Endpoints

- **POST /api/classify**
  - Input: Image file (`image` field)
  - Output: JSON with `class_id`, `class_name`, and `confidence`
  - Example:

    ```bash
    curl -X POST -F "image=@image.jpg" http://localhost:5000/api/classify
    ```

    Response:

    ```json
    {"class_id": 1, "class_name": "goldfish", "confidence": 0.9234}
    ```

- **POST /api/explain**
  - Input: Image file (`image` field) and class ID (`class_id` field)
  - Output: JSON with base64-encoded explanation image (`explanation`)
  - Example:

    ```bash
    curl -X POST -F "image=@image.jpg" -F "class_id=1" http://localhost:5000/api/explain
    ```

    Response:

    ```json
    {"explanation": "<base64_string>"}
    ```

## Dependencies

Listed in `requirements.txt`:

- flask
- pillow
- captum
- gunicorn
- torchvision
- werkzeug

## Configuration

Settings are defined in `app/config.py`:

- `SECRET_KEY`: Set via environment variable (`SECRET_KEY`) for security.
- `UPLOAD_FOLDER`: `static/uploads` for temporary image storage.
- `MAX_CONTENT_LENGTH`: 16MB upload limit.
- `ALLOWED_EXTENSIONS`: `.jpg`, `.jpeg`, `.png`.

## Notes

- **Production:** Use `gunicorn` (configured in Dockerfile) instead of Flask's development server.
- **GPU Support:** The Dockerfile uses a CUDA-enabled PyTorch image. For CPU-only, modify to `pytorch/pytorch:1.9.0-cpu`.
- **Windows Users:** Ensure Docker Desktop is running. If permission issues occur with `static/uploads`, add `RUN chmod -R 777 /app/static/uploads` to the Dockerfile.
- **Extending the App:** Add new models or explanation methods by extending `app/services/`.

## Troubleshooting

- **404 for favicon.ico:** Add a favicon to `app/static/` or remove the favicon route.
- **Permission errors:** Check write permissions for `static/uploads`.
- **API errors:** Ensure image files are valid and `class_id` is an integer.

## License

MIT License (or specify your preferred license).
