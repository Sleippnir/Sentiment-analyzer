import argparse
from sentiment_analysis import analyze_sentiment_combined
from utils import MAX_INPUT_LENGTH
import logging
import asyncio
from transformers import pipeline

logging.basicConfig(level=logging.INFO)

def cli(transformers_pipeline):
    parser = argparse.ArgumentParser(description="Sentiment Analysis Tool")
    parser.add_argument("text", type=str, help="Input text to analyze")
    args = parser.parse_args()

    if len(args.text) > MAX_INPUT_LENGTH:
        logging.warning(f"Input exceeds max length of {MAX_INPUT_LENGTH}")
        print(f"Error: Input text exceeds {MAX_INPUT_LENGTH} characters")
        return

    try:
        result = asyncio.run(analyze_sentiment_combined(args.text, transformers_pipeline))
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print("\nSentiment Analysis Results:")
            print(f"- Text: {result['text']}")
            print(f"- TextBlob: {result['textblob']}")
            print(f"- NLTK: {result['nltk']}")
            print(f"- Transformers: {result['transformers']['label']} "
                  f"(Confidence: {result['transformers']['confidence']:.2f})\n")
    except Exception as e:
        logging.exception("CLI execution failed")
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    transformers_pipeline = pipeline("sentiment-analysis")
    cli(transformers_pipeline)
