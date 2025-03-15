import unittest
from unittest.mock import patch
from module5 import main
import sys

class TestEntryPoint(unittest.TestCase):
    @patch("module5.cli")
    def test_cli_mode(self, mock_cli):
        with patch.object(sys, "argv", ["entry_point.py", "I love this product!"]):
            main()
            mock_cli.assert_called_once()

    @patch("module5.app.run")
    def test_api_mode(self, mock_run):
        with patch.object(sys, "argv", ["entry_point.py"]):
            main()
            mock_run.assert_called_once()

    @patch("module5.load_ssl_context")
    def test_missing_ssl_certificates(self, mock_load_ssl):
        mock_load_ssl.return_value = None
        with patch.object(sys, "argv", ["entry_point.py"]):
            with self.assertLogs(level="ERROR") as log:
                main()
                self.assertIn("Failed to load SSL certificates", log.output[0])

if __name__ == '__main__':
    unittest.main()
