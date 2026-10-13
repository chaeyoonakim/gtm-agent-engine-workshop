import copy
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import data_service, gtm_agent
from gtm_agent.gtm_records import PROSPECTS


class ProspectUpdateTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = next(iter(PROSPECTS))
        self.original_record = copy.deepcopy(PROSPECTS[self.prospect_id])
        self.cached_profile = data_service._PROFILES.get(self.prospect_id)
        data_service._PROFILES.pop(self.prospect_id, None)

    def tearDown(self):
        PROSPECTS[self.prospect_id] = self.original_record
        if self.cached_profile is None:
            data_service._PROFILES.pop(self.prospect_id, None)
        else:
            data_service._PROFILES[self.prospect_id] = self.cached_profile

    def test_update_persists_and_returns_saved_stack(self):
        technology = "NewTechnology"
        data_service._PROFILES[self.prospect_id] = {"tech_stack": []}

        result = data_service.update_prospect_info(self.prospect_id, technology)

        self.assertEqual(PROSPECTS[self.prospect_id]["tech_stack"], result["tech_stack"])
        self.assertIn(technology, result["tech_stack"])
        self.assertNotIn(self.prospect_id, data_service._PROFILES)

    def test_cached_profile_refreshes_stack_and_preserves_other_fields(self):
        profile = {"prospect_id": self.prospect_id, "company": "Acme", "tech_stack": []}
        data_service._PROFILES[self.prospect_id] = profile
        PROSPECTS[self.prospect_id]["tech_stack"] = ["LaterTechnology"]

        result = gtm_agent.build_prospect_profile.func(self.prospect_id)["prospect_profile"]

        self.assertEqual(result["tech_stack"], ["LaterTechnology"])
        self.assertEqual(result["company"], "Acme")
        self.assertEqual(data_service._PROFILES[self.prospect_id], result)

    def test_update_then_score_uses_saved_stack_and_corrects_conflict(self):
        technology = "NewTechnology"
        update = data_service.update_prospect_info(self.prospect_id, technology)
        profile = {"prospect_id": self.prospect_id, "tech_stack": self.original_record["tech_stack"]}
        offering = {
            "required_tech_stack": [technology, "MissingTechnology"],
            "min_annual_revenue": 1,
            "description": "Test offering",
        }
        scorer_response = {
            "score": 55.0,
            "max_score": 100,
            "justification": "NewTechnology is missing, so the stack alignment is poor.",
            "rubric_breakdown": {
                "revenue_fit": 70.0,
                "tech_stack_match": 0.0,
                "segment_fit": 70.0,
                "component_max": 100,
            },
        }
        scorer_result = SimpleNamespace(model_dump=lambda: copy.deepcopy(scorer_response))

        scorer = SimpleNamespace(invoke=unittest.mock.Mock(return_value=scorer_result))
        with patch.object(gtm_agent, "_scoring_llm", scorer):
            result = gtm_agent.score_prospect.func(profile, offering)

        scoring_input = json.loads(scorer.invoke.call_args.args[0][1]["content"].split("\n\nProspect profile:\n")[1])
        self.assertIn(technology, update["tech_stack"])
        self.assertIn(technology, scoring_input["tech_stack"])
        self.assertNotIn("NewTechnology is missing", result["justification"])
        self.assertIn("Required technologies present: NewTechnology.", result["justification"])
        self.assertIn("Required technologies missing: MissingTechnology.", result["justification"])
        self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], 50.0)
        self.assertGreater(result["score"], scorer_response["score"])


if __name__ == "__main__":
    unittest.main()
