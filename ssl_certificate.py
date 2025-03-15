from api import app
from cli import cli
import sys
import logging
from decouple import config
from ssl_certificate import load_ssl_context
from transformers import pipeline

def main():
    """
    Entry point for the application.
    Runs the CLI if arguments are provided, otherwise starts the Flask server.

    The application supports two modes:
    1. **CLI Mode**: Users can input text via the command line and receive sentiment analysis results.
    2. **API Mode**: Users can send text to a REST API and receive sentiment analysis results in JSON format.

    Environment Variables:
        LOG_LEVEL (str): Logging level (e.g., "INFO", "DEBUG"). Defaults to "INFO".
        HOST (str): Host address for the Flask server. Defaults to "0.0.0.0".
        PORT (int): Port number for the Flask server. Defaults to 5000.
        SSL_CERT_PATH (str): Path to the SSL certificate file. Defaults to "cert.pem".
        SSL_KEY_PATH (str): Path to the SSL key file. Defaults to "key.pem".

    Example Usage:
        # Run CLI mode
        python entry_point.py "I love this product!"

        # Run API mode
        python entry_point.py
    """
    # Configure logging
    logging.basicConfig(level=config("LOG_LEVEL", default="INFO"), format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Initialize the transformers pipeline
    transformers_pipeline = pipeline("sentiment-analysis")

    logger.info("Starting application...")
    if len(sys.argv) > 1:  # If arguments are provided, run CLI
        cli(transformers_pipeline)
    else:  # Otherwise, start Flask server
        ssl_context = load_ssl_context()
        if ssl_context is None:
            logger.error("Failed to load SSL certificates. Exiting...")
            return

        app.run(
            ssl_context=ssl_context,
            host=config("HOST", default="0.0.0.0"),
            port=config("PORT", default=5000, cast=int)
        )

if __name__ == '__main__':
    main()
