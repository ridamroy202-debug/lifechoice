import os
import unittest

from app.settings import _normalize_env_value, env_str, load_environment


class SettingsTests(unittest.TestCase):
    def test_normalize_env_value_strips_whitespace_and_quotes(self):
        self.assertEqual(_normalize_env_value("  value  "), "value")
        self.assertEqual(_normalize_env_value(' "value" '), "value")
        self.assertEqual(_normalize_env_value(" 'value' "), "value")

    def test_env_str_returns_trimmed_runtime_value(self):
        previous = os.environ.get("TEST_SETTINGS_VALUE")
        try:
            os.environ["TEST_SETTINGS_VALUE"] = "  sample-value  "
            load_environment()
            self.assertEqual(env_str("TEST_SETTINGS_VALUE"), "sample-value")
        finally:
            if previous is None:
                os.environ.pop("TEST_SETTINGS_VALUE", None)
            else:
                os.environ["TEST_SETTINGS_VALUE"] = previous


if __name__ == "__main__":
    unittest.main()
