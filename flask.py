from flask import Flask, request, jsonify, abort
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flasgger import Swagger
from sentiment_analysis import analyze_sentiment_combined
from utils import MAX_INPUT_LENGTH
from ssl_certificate import load_ssl_context
import logging
from decouple import config
from transformers import pipeline

app = Flask(__name__)

# Initialize Swagger for API documentation
Swagger(app, template={
    "swagger": "2.0",
    "info": {
        "title": "Sentiment Analysis API",
        "description": "An API for analyzing the sentiment of text using TextBlob, NLTK, and Hugging Face Transformers.",
        "version": "1.0.0"
    },
    "basePath": "/",
    "schemes": ["https"],
    "consumes": ["application/json"],
    "produces": ["application/json"]
})

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,  # Use client IP address for rate limiting
    default_limits=[config("RATE_LIMIT", default="10 per minute")]  # Configurable rate limit
)

@app.before_request
def enforce_global_timeout():
    """
    Set a global timeout for all requests.
    """
    request.environ['REQUEST_TIMEOUT'] = config("REQUEST_TIMEOUT", default=15, cast=int)  # Configurable timeout

@app.route('/analyze', methods=['POST'])
@limiter.limit(config("RATE_LIMIT", default="10 per minute"))  # Apply rate limit to this endpoint
async def analyze_sentiment_api():
    """
    Analyze the sentiment of the input text.
    ---
    tags:
      - Sentiment Analysis
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            text:
              type: string
              description: The text to analyze.
              example: "I love this product!"
    responses:
      200:
        description: Sentiment analysis results.
        schema:
          type: object
          properties:
            text:
              type: string
              description: The sanitized input text.
            textblob:
              type: string
              description: Sentiment result from TextBlob.
            nltk:
              type: string
              description: Sentiment result from NLTK.
            transformers:
              type: object
              properties:
                label:
                  type: string
                  description: Sentiment label from Transformers.
                confidence:
                  type: number
                  description: Confidence score from Transformers.
      400:
        description: Invalid input or missing text.
      500:
        description: Internal server error.
    """
    try:
        data = request.get_json(force=True)  # Force parsing even if Content-Type is not application/json
    except Exception as e:
        logging.error(f"Invalid JSON payload: {e}")
        abort(400, description={"error": "Invalid JSON payload"})

    if not data or 'text' not in data:
        logging.warning("No text provided in API request")
        abort(400, description={"error": "No text provided"})
    
    text = data['text']
    if not isinstance(text, str):
        logging.warning("Invalid input text type in API request")
        abort(400, description={"error": "Invalid input text"})
    
    if len(text) > MAX_INPUT_LENGTH:
        logging.warning(f"Input text exceeds maximum length of {MAX_INPUT_LENGTH} characters")
        abort(400, description={"error": f"Input text exceeds maximum length of {MAX_INPUT_LENGTH} characters"})
    
    try:
        result = await analyze_sentiment_combined(text, transformers_pipeline)
        return jsonify(result)
    except TimeoutError:
        logging.error("API request timed out")
        abort(504, description={"error": "Request timed out"})
    except Exception as e:
        logging.error(f"Unexpected error in API: {e}")
        abort(500, description={"error": "Internal server error"})

def start_flask_server():
    """
    Start the Flask server with SSL certificates loaded from Module 6.
    """
    ssl_context = load_ssl_context()
    if ssl_context is None:
        logging.error("Failed to load SSL certificates. Exiting...")
        return

    app.run(
        ssl_context=ssl_context,
        host=config("HOST", default="0.0.0.0"),
        port=config("PORT", default=5000, cast=int)
    )

if __name__ == '__main__':
    start_flask_server()
