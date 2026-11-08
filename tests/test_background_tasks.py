import unittest
from unittest.mock import Mock

from background_tasks import make_task_key, submit_task


class BackgroundTaskTests(unittest.TestCase):
    def test_same_key_reuses_in_progress_task(self):
        function = Mock(return_value="ready")
        key = make_task_key("test-dedup", "same")

        first = submit_task(key, function)
        second = submit_task(key, function)

        self.assertIs(first, second)
        self.assertEqual(first.result(timeout=2), "ready")
        function.assert_called_once_with()

    def test_failed_task_can_be_retried_explicitly(self):
        function = Mock(side_effect=[RuntimeError("first"), "second"])
        key = make_task_key("test-retry", "failure")

        first = submit_task(key, function)
        with self.assertRaises(RuntimeError):
            first.result(timeout=2)

        second = submit_task(key, function, retry_failed=True)
        self.assertIsNot(first, second)
        self.assertEqual(second.result(timeout=2), "second")
        self.assertEqual(function.call_count, 2)

    def test_task_keys_are_opaque_and_content_sensitive(self):
        first = make_task_key("summary", "document one", "model")
        second = make_task_key("summary", "document two", "model")
        self.assertNotEqual(first, second)
        self.assertTrue(all(character in "0123456789abcdef" for character in first))


if __name__ == "__main__":
    unittest.main()
