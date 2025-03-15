import unittest
from unittest.mock import patch
from module3 import app
import json

class TestFlaskAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    @patch("module3.analyze_sentiment_combined")
    def test_analyze_endpoint_valid(self, mock_analyze):
        mock_analyze.return_value = {
            "text": "I love this product!",
            "textblob": "Positive",
            "nltk": "Positive",
            "transformers": {"label": "Positive", "confidence": 0.95}
        }

        response = self.client.post(
            "/analyze",
            data=json.dumps({"text": "I love this product!"}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Positive", response.get_data(as_text=True))

    def test_analyze_endpoint_invalid(self):
        response = self.client.post(
            "/analyze",
            data=json.dumps({}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)

    @patch("module3.analyze_sentiment_combined")
    def test_analyze_endpoint_error(self, mock_analyze):
        mock_analyze.side_effect = Exception("Internal error")
        response = self.client.post(
            "/analyze",
            data=json.dumps({"text": "I love this product!"}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 500)

    def test_analyze_endpoint_large_input(self):
        large_text = "a" * (10000 - 1)  # Just below MAX_INPUT_LENGTH
        response = self.client.post(
            "/analyze",
            data=json.dumps({"text": large_text}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
