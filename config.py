import os
from pathlib import Path

class Config:
    """Application configuration class."""
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key")
    UPLOAD_FOLDER = Path("static/uploads")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB limit
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}