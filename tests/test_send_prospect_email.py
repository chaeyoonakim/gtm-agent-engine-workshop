import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gtm_agent import gtm_agent


class SendProspectEmailTests(unittest.TestCase):
    def call_tool(self, prospect):
        runtime = SimpleNamespace(config={"metadata": {}})
        return gtm_agent.send_prospect_email.func(
            prospect=prospect,
            subject="A helpful introduction",
            body="Hello",
            runtime=runtime,
            from_rep={},
        )

    @patch("gtm_agent.gtm_agent.uuid.uuid4")
    @patch("gtm_agent.gtm_agent.data_service.get_prospect_record")
    def test_disqualified_prospect_is_blocked_without_sending(self, get_record, make_uuid):
        get_record.return_value = {
            "prospect_id": "LEAD-50001",
            "name": "Source Prospect",
            "email": "source@example.com",
            "disqualified": True,
        }

        result = self.call_tool({
            "prospect_id": "LEAD-50001",
            "name": "Model Prospect",
            "email": "model@example.com",
        })

        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect is disqualified; email not sent.",
            "prospect_id": "LEAD-50001",
        })
        make_uuid.assert_not_called()

    @patch("gtm_agent.gtm_agent.uuid.uuid4")
    @patch("gtm_agent.gtm_agent.data_service.get_prospect_record")
    def test_qualified_prospect_uses_source_email(self, get_record, make_uuid):
        get_record.return_value = {
            "prospect_id": "LEAD-50002",
            "name": "Source Prospect",
            "email": "source@example.com",
            "disqualified": False,
        }
        make_uuid.return_value.hex = "1234567890abcdef"

        result = self.call_tool({
            "prospect_id": "LEAD-50002",
            "name": "Model Prospect",
            "email": "model@example.com",
        })

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], "source@example.com")
        self.assertEqual(result["to_name"], "Source Prospect")

    @patch("gtm_agent.gtm_agent.uuid.uuid4")
    @patch("gtm_agent.gtm_agent.data_service.get_prospect_record", return_value=None)
    def test_missing_source_record_fails_without_sending(self, get_record, make_uuid):
        result = self.call_tool({"prospect_id": "LEAD-MISSING", "email": "model@example.com"})

        self.assertEqual(result, {
            "status": "failed",
            "error": "Prospect record not found.",
            "prospect_id": "LEAD-MISSING",
        })
        make_uuid.assert_not_called()

    def test_system_prompt_requires_disqualification_check(self):
        prompt = gtm_agent.SYSTEM_PROMPT.lower()

        self.assertIn("call get_prospect", prompt)
        self.assertIn("check its disqualified flag", prompt)
        self.assertIn("never email a disqualified prospect", prompt)
        self.assertIn("tell the rep", prompt)
        self.assertNotIn("always carry the request through", prompt)
        self.assertNotIn("complete the email the rep asked for every time", prompt)


if __name__ == "__main__":
    unittest.main()
