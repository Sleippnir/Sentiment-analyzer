import argparse
from sentiment_analysis import analyze_sentiment_combined
from utils import MAX_INPUT_LENGTH
import logging
import asyncio
from transformers import pipeline

def cli(transformers_pipeline):
    """
    Command-line interface (CLI) for sentiment analysis.

    This CLI allows users to input text and receive sentiment analysis results
    from TextBlob, NLTK, and Hugging Face Transformers.

    Usage:
        python cli.py "I love this product!"

    Args:
        transformers_pipeline (Pipeline): A Hugging Face Transformers pipeline for sentiment analysis.

    Environment Variables:
        MAX_INPUT_LENGTH (int): Maximum allowed length for input text. Defaults to 10000 characters.

    Output:
        Prints the sentiment analysis results in a user-friendly format:
        - Text: The sanitized input text.
        - TextBlob: Sentiment result from TextBlob.
        - NLTK: Sentiment result from NLTK.
        - Transformers: Sentiment label and confidence score from Hugging Face Transformers.

    Example:
        $ python cli.py "I love this product!"
        Sentiment Analysis Results:
        - Text: I love this product!
        - TextBlob: Positive
        - NLTK: Positive
        - Transformers: Positive (Confidence: 0.95)
    """
    parser = argparse.ArgumentParser(description="Sentiment Analysis Tool")
    parser.add_argument("text", type=str, help="Input text to analyze")
    args = parser.parse_args()
    
    # Validate input length
    if len(args.text) > MAX_INPUT_LENGTH:
        logging.warning(f"Input text exceeds maximum length of {MAX_INPUT_LENGTH} characters")
        print(f"Error: Input text exceeds maximum length of {MAX_INPUT_LENGTH} characters")
        return

    try:
        # Perform sentiment analysis
        result = asyncio.run(analyze_sentiment_combined(args.text, transformers_pipeline))
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            # Print results in a user-friendly format
            print("\nSentiment Analysis Results:")
            print(f"- Text: {result['text']}")
            print(f"- TextBlob: {result['textblob']}")
            print(f"- NLTK: {result['nltk']}")
            print(f"- Transformers: {result['transformers']['label']} (Confidence: {result['transformers']['confidence']:.2f})\n")
    except Exception as e:
        print(f"Error: {e}")
