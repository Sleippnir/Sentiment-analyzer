import asyncio
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from functools import lru_cache
from utils import sanitize_input, detect_language, MAX_INPUT_LENGTH
import logging

@lru_cache(maxsize=100)  # Cache up to 100 results
async def analyze_sentiment_combined(text, transformers_pipeline):
    """
    Analyzes sentiment using TextBlob, NLTK, and Transformers concurrently.

    Args:
        text (str): The input text to analyze.
        transformers_pipeline (Pipeline): A Hugging Face Transformers pipeline for sentiment analysis.

    Returns:
        dict: A dictionary containing the sanitized text and sentiment analysis results from TextBlob, NLTK, and Transformers.
              Example:
              {
                  "text": "I love this product!",
                  "textblob": "Positive",
                  "nltk": "Positive",
                  "transformers": {"label": "Positive", "confidence": 0.95}
              }

    Raises:
        Exception: If sentiment analysis fails for all methods.
    """
    sanitized_text = sanitize_input(text)
    if not sanitized_text:
        return {"error": "Invalid or empty input text"}

    # Detect language
    if not detect_language(sanitized_text):
        return {"error": "Unsupported language. Only English, Spanish, and French are supported."}

    try:
        # Run TextBlob, NLTK, and Transformers sentiment analysis concurrently
        textblob_result, nltk_result, transformers_result = await asyncio.gather(
            get_textblob_sentiment(sanitized_text),
            get_nltk_sentiment(sanitized_text),
            get_transformers_sentiment(sanitized_text, transformers_pipeline)
        )
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return {"error": "Sentiment analysis failed"}

    return {
        "text": sanitized_text,
        "textblob": textblob_result,
        "nltk": nltk_result,
        "transformers": {
            "label": transformers_result[0],
            "confidence": transformers_result[1]
        }
    }

async def get_textblob_sentiment(text):
    """
    Analyzes sentiment using TextBlob.

    Args:
        text (str): The input text to analyze.

    Returns:
        str: The sentiment label ("Positive", "Negative", or "Neutral").

    Raises:
        Exception: If TextBlob sentiment analysis fails.
    """
    try:
        blob = TextBlob(text)
        sentiment_polarity = blob.sentiment.polarity
        if sentiment_polarity > 0:
            return "Positive"
        elif sentiment_polarity < 0:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        logger.error(f"TextBlob sentiment analysis failed: {e}")
        return "Error"

async def get_nltk_sentiment(text):
    """
    Analyzes sentiment using NLTK's VADER sentiment analyzer.

    Args:
        text (str): The input text to analyze.

    Returns:
        str: The sentiment label ("Positive", "Negative", or "Neutral").

    Raises:
        Exception: If NLTK sentiment analysis fails.
    """
    try:
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = sia.polarity_scores(text)
        if sentiment_scores['compound'] >= 0.05:
            return "Positive"
        elif sentiment_scores['compound'] <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        logger.error(f"NLTK sentiment analysis failed: {e}")
        return "Error"

async def get_transformers_sentiment(text, transformers_pipeline):
    """
    Analyzes sentiment using a Hugging Face Transformers pipeline.

    Args:
        text (str): The input text to analyze.
        transformers_pipeline (Pipeline): A Hugging Face Transformers pipeline for sentiment analysis.

    Returns:
        tuple: A tuple containing the sentiment label and confidence score.
               Example: ("Positive", 0.95)

    Raises:
        Exception: If Transformers sentiment analysis fails.
    """
    try:
        result = transformers_pipeline(text)[0]
        return result['label'].capitalize(), result['score']
    except Exception as e:
        logger.error(f"Transformers sentiment analysis failed: {e}")
        return "Neutral", 0.0  # Fallback result
