import unittest
from unittest.mock import patch, MagicMock
from io import StringIO
import sys
from module4 import cli

class TestCLI(unittest.TestCase):
    @patch("module4.analyze_sentiment_combined")
    def test_cli_valid_input(self, mock_analyze):
        mock_analyze.return_value = {
            "text": "I love this product!",
            "textblob": "Positive",
            "nltk": "Positive",
            "transformers": {"label": "Positive", "confidence": 0.95}
        }

        with patch.object(sys, "argv", ["cli.py", "I love this product!"]):
            with patch("sys.stdout", new=StringIO()) as fake_output:
                cli(MagicMock())
                output = fake_output.getvalue()
                self.assertIn("Positive", output)

    def test_cli_input_exceeds_length(self):
        long_text = "a" * 15000
        with patch.object(sys, "argv", ["cli.py", long_text]):
            with patch("sys.stdout", new=StringIO()) as fake_output:
                cli(MagicMock())
                output = fake_output.getvalue()
                self.assertIn("exceeds maximum length", output)

    @patch("module4.analyze_sentiment_combined")
    def test_cli_error_handling(self, mock_analyze):
        mock_analyze.side_effect = Exception("Sentiment analysis failed")
        with patch.object(sys, "argv", ["cli.py", "I love this product!"]):
            with patch("sys.stdout", new=StringIO()) as fake_output:
                cli(MagicMock())
                output = fake_output.getvalue()
                self.assertIn("Error", output)

    def test_cli_special_characters(self):
        special_char_text = "I ❤️ this product!"
        with patch.object(sys, "argv", ["cli.py", special_char_text]):
            with patch("sys.stdout", new=StringIO()) as fake_output:
                cli(MagicMock())
                output = fake_output.getvalue()
                self.assertIn("Text:", output)

if __name__ == '__main__':
    unittest.main()
