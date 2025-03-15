import unittest
from unittest.mock import patch
from module1 import sanitize_input, detect_language

class TestInputSanitization(unittest.TestCase):
    def test_sanitize_input_valid(self):
        self.assertEqual(sanitize_input("<html>I love this product!</html>"), "I love this product!")

    def test_sanitize_input_invalid(self):
        self.assertIsNone(sanitize_input(None))
        self.assertIsNone(sanitize_input(""))

    def test_sanitize_input_length(self):
        long_text = "a" * 15000
        self.assertEqual(len(sanitize_input(long_text)), 10000)

    def test_sanitize_input_non_ascii(self):
        non_ascii_text = "こんにちは"  # Japanese greeting
        self.assertEqual(sanitize_input(non_ascii_text), non_ascii_text)

    @patch("module1.config")
    def test_detect_language_supported(self, mock_config):
        mock_config.return_value = "en,es,fr"
        self.assertTrue(detect_language("I love this product!"))  # English
        self.assertTrue(detect_language("Me encanta este producto!"))  # Spanish

    @patch("module1.config")
    def test_detect_language_unsupported(self, mock_config):
        mock_config.return_value = "en,es,fr"
        self.assertFalse(detect_language("Ciao, come stai?"))  # Italian

    @patch("module1.config")
    def test_detect_language_failure(self, mock_config):
        mock_config.return_value = "en,es,fr"
        self.assertTrue(detect_language("1234567890"))  # Fallback

if __name__ == '__main__':
    unittest.main()
