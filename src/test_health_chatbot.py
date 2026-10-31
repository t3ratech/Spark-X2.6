"""Tests for the health chatbot.

Uses only the Python standard library (unittest), so no ML dependencies or
model checkpoints are required to run these tests.
"""

import unittest

from health_chatbot import build_messages, SYSTEM_PROMPT


class BuildMessagesTest(unittest.TestCase):
    def test_builds_system_and_user_messages(self):
        messages = build_messages("I have a fever. What should I do?")
        self.assertIsInstance(messages, list)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("fever", messages[1]["content"])

    def test_system_prompt_keeps_clinician_boundary(self):
        self.assertIn("clinician", SYSTEM_PROMPT.lower())
        self.assertIn("diagnos", SYSTEM_PROMPT.lower())
        self.assertIn("treatment", SYSTEM_PROMPT.lower())

    def test_build_messages_includes_user_question(self):
        question = "Can exercise help lower blood pressure?"
        messages = build_messages(question)
        self.assertIn(question, messages[1]["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
