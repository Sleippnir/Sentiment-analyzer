import re
import logging
from langdetect import detect, LangDetectException
from decouple import config

# Configure logging globally
logging.basicConfig(level=config("LOG_LEVEL", default="INFO"), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
MAX_INPUT_LENGTH = config("MAX_INPUT_LENGTH", default=10000, cast=int)
SUPPORTED_LANGUAGES = config("SUPPORTED_LANGUAGES", default="en,es,fr").split(",")  # Configurable supported languages
STRICT_LANGUAGE_CHECK = config("STRICT_LANGUAGE_CHECK", default=False, cast=bool)  # Configurable strict language check

def sanitize_input(text):
    """
    Sanitizes the input text by:
    1. Removing HTML tags.
    2. Escaping special characters.
    3. Trimming whitespace.
    4. Limiting input length.

    Args:
        text (str): The input text to sanitize.

    Returns:
        str: The sanitized text, or None if the input is invalid or empty.
    """
    if not text or not isinstance(text, str) or text.strip() == "":
        logger.warning("Invalid or empty input text provided")
        return None

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Escape special characters (optional, depending on use case)
    text = text.replace('\n', ' ').replace('\r', ' ').strip()

    # Limit input length to prevent abuse
    if len(text) > MAX_INPUT_LENGTH:
        logger.warning(f"Input text truncated to {MAX_INPUT_LENGTH} characters")
        text = text[:MAX_INPUT_LENGTH]

    return text

def detect_language(text):
    """
    Detects the language of the input text.
    Returns True if the language is supported, False otherwise.
    If strict language checks are disabled, returns True as a fallback.

    Args:
        text (str): The input text to analyze.

    Returns:
        bool: True if the language is supported or strict checks are disabled, False otherwise.

    Environment Variables:
        SUPPORTED_LANGUAGES (str): A comma-separated list of supported language codes.
                                   Defaults to "en,es,fr" (English, Spanish, French).
                                   Example: "en,es,fr,de" to add German.
        STRICT_LANGUAGE_CHECK (bool): If True, rejects unsupported languages. If False, allows processing as a fallback.
                                      Defaults to False.
    """
    try:
        language = detect(text)
        if language not in SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language detected: {language}")
            if STRICT_LANGUAGE_CHECK:
                return False  # Strict check: reject unsupported languages
            else:
                logger.info("Strict language check is disabled. Allowing processing as a fallback.")
                return True  # Fallback: allow processing even if language is unsupported
        return True
    except LangDetectException as e:
        logger.error(f"Language detection failed: {e}")
        if STRICT_LANGUAGE_CHECK:
            return False  # Strict check: reject if language detection fails
        else:
            logger.info("Strict language check is disabled. Allowing processing as a fallback.")
            return True  # Fallback: allow processing if language detection fails
