from flask import Flask, request, jsonify, abort, make_response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flasgger import Swagger
from sentiment_analysis import analyze_sentiment_combined
from utils import MAX_INPUT_LENGTH
from ssl_certificate import load_ssl_context
from transformers import pipeline
from decouple import config
import logging
import asyncio

# ----------------------------- #
# App Initialization
# ----------------------------- #

app = Flask(__name__)

# Swagger API Documentation
Swagger(app, template={
    "swagger": "2.0",
    "info": {
        "title": "Sentiment Analysis API",
        "description": "Analyze sentiment using TextBlob, NLTK, and Transformers.",
        "version": "1.0.0"
    },
    "basePath": "/",
    "schemes": ["https"],
    "consumes": ["application/json"],
    "produces": ["application/json"]
})

# Rate Limiting Setup
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[config("RATE_LIMIT", default="10 per minute")]
)
limiter.init_app(app)

# Load sentiment model
transformers_pipeline = pipeline("sentiment-analysis")

# ----------------------------- #
# Middleware
# ----------------------------- #

@app.before_request
def enforce_global_timeout():
    request.environ['REQUEST_TIMEOUT'] = config("REQUEST_TIMEOUT", default=15, cast=int)

# ----------------------------- #
# Routes
# ----------------------------- #

@app.route('/analyze', methods=['POST'])
@limiter.limit(config("RATE_LIMIT", default="10 per minute"))
def analyze_sentiment_api():
    """
    Analyze sentiment from input text.
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
              example: "I love this product!"
    responses:
      200:
        description: Sentiment analysis results.
        schema:
          type: object
          properties:
            text:
              type: string
            textblob:
              type: string
            nltk:
              type: string
            transformers:
              type: object
              properties:
                label:
                  type: string
                confidence:
                  type: number
      400:
        description: Invalid request.
      500:
        description: Internal server error.
    """
    try:
        data = request.get_json(force=True)
    except Exception as e:
        logging.error(f"Invalid JSON payload: {e}")
        abort(make_response(jsonify(error="Invalid JSON payload"), 400))

    if not data or 'text' not in data:
        logging.warning("Missing 'text' in request")
        abort(make_response(jsonify(error="Missing 'text' in request"), 400))

    text = data['text']

    if not isinstance(text, str):
        logging.warning("Invalid type for 'text'")
        abort(make_response(jsonify(error="'text' must be a string"), 400))

    if len(text) > MAX_INPUT_LENGTH:
        logging.warning("Text too long")
        abort(make_response(jsonify(error=f"Text exceeds {MAX_INPUT_LENGTH} characters"), 400))

    try:
        result = asyncio.run(analyze_sentiment_combined(text, transformers_pipeline))
        return jsonify(result)
    except TimeoutError:
        logging.error("Request timed out")
        abort(make_response(jsonify(error="Request timed out"), 504))
    except Exception as e:
        logging.exception("Unexpected error during analysis")
        abort(make_response(jsonify(error="Internal server error"), 500))

# ----------------------------- #
# Entry Point
# ----------------------------- #

def start_flask_server():
    ssl_context = load_ssl_context()
    if ssl_context is None:
        logging.error("Failed to load SSL certificates.")
        return

    app.run(
        ssl_context=ssl_context,
        host=config("HOST", default="0.0.0.0"),
        port=config("PORT", default=5000, cast=int)
    )

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    start_flask_server()
