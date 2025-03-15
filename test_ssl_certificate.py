import unittest
from unittest.mock import patch
from module6 import load_ssl_context
import os

class TestSSLCertificate(unittest.TestCase):
    @patch("module6.os.path.exists")
    def test_load_ssl_context_valid(self, mock_exists):
        mock_exists.return_value = True
        result = load_ssl_context()
        self.assertIsNotNone(result)

    @patch("module6.os.path.exists")
    def test_load_ssl_context_missing(self, mock_exists):
        mock_exists.return_value = False
        result = load_ssl_context()
        self.assertIsNone(result)

    @patch("module6.os.path.exists")
    def test_load_ssl_context_invalid_format(self, mock_exists):
        mock_exists.return_value = True
        with patch("module6.open", side_effect=IOError("Invalid format")):
            result = load_ssl_context()
            self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
