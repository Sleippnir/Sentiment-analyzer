import unittest
from unittest.mock import patch, MagicMock
from module2 import analyze_sentiment_combined, get_textblob_sentiment, get_nltk_sentiment, get_transformers_sentiment
import asyncio

class TestSentimentAnalysis(unittest.TestCase):
    @patch("module2.get_textblob_sentiment")
    @patch("module2.get_nltk_sentiment")
    @patch("module2.get_transformers_sentiment")
    async def test_analyze_sentiment_combined_valid(self, mock_transformers, mock_nltk, mock_textblob):
        mock_textblob.return_value = "Positive"
        mock_nltk.return_value = "Positive"
        mock_transformers.return_value = ("Positive", 0.95)

        result = await analyze_sentiment_combined("I love this product!", MagicMock())
        self.assertEqual(result["textblob"], "Positive")
        self.assertEqual(result["nltk"], "Positive")
        self.assertEqual(result["transformers"]["label"], "Positive")

    @patch("module2.get_textblob_sentiment")
    @patch("module2.get_nltk_sentiment")
    @patch("module2.get_transformers_sentiment")
    async def test_analyze_sentiment_combined_error(self, mock_transformers, mock_nltk, mock_textblob):
        mock_textblob.side_effect = Exception("TextBlob failed")
        mock_nltk.side_effect = Exception("NLTK failed")
        mock_transformers.side_effect = Exception("Transformers failed")

        result = await analyze_sentiment_combined("I love this product!", MagicMock())
        self.assertEqual(result["error"], "Sentiment analysis failed")

    async def test_get_textblob_sentiment_positive(self):
        result = await get_textblob_sentiment("I love this product!")
        self.assertEqual(result, "Positive")

    async def test_get_nltk_sentiment_positive(self):
        result = await get_nltk_sentiment("I love this product!")
        self.assertEqual(result, "Positive")

    async def test_get_transformers_sentiment_positive(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "POSITIVE", "score": 0.95}]
        result = await get_transformers_sentiment("I love this product!", mock_pipeline)
        self.assertEqual(result, ("Positive", 0.95))

    async def test_analyze_sentiment_combined_short_input(self):
        result = await analyze_sentiment_combined("!", MagicMock())
        self.assertIn("textblob", result)  # Ensure the result is processed

if __name__ == '__main__':
    unittest.main()
