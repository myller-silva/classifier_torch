"""Flask application factory for the web application."""

import os
from flask import Flask
from config import Config
from routes.web import web_bp
from routes.api import api_bp


def create_app():
    """Factory function to create and configure the Flask app."""
    app = Flask(__name__)
    app.config.from_object(Config)

    # Ensure upload folder exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Register blueprints
    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    return app


# Initialize the app
app = create_app()

if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True, host="0.0.0.0", port=5000)
