import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gtm_agent import gtm_agent
from gtm_agent import data_service


PROSPECT_ID = "LEAD-TEST"
BILLING_QUALIFICATION = {
    "tax_id": "123-45-6789",
    "date_of_birth": "1980-01-01",
    "card_on_file": "4111111111111111",
    "credit_check_ref": "EXPN-TEST",
}


class ProspectDataFilteringTests(unittest.TestCase):
    def profile_cache(self):
        profiles = data_service._PROFILES
        previous_profile = profiles.pop(PROSPECT_ID, None)
        self.addCleanup(profiles.pop, PROSPECT_ID, None)
        if previous_profile is not None:
            self.addCleanup(profiles.__setitem__, PROSPECT_ID, previous_profile)
        return profiles

    def patch_data_service(self, name, replacement):
        service_patch = patch.object(data_service, name, replacement)
        service_patch.start()
        self.addCleanup(service_patch.stop)

    def test_get_prospect_returns_only_available_allow_listed_fields(self):
        record = {
            "name": "Taylor Prospect",
            "email": "taylor@example.com",
            "disqualified": False,
            "annual_revenue": 5000000,
            "enrichment_source": "CRM",
            "billing_qualification": BILLING_QUALIFICATION,
            "engagement_history": ["opened email"],
            "unlisted_field": "not for the agent",
        }
        self.patch_data_service("get_prospect_record", lambda _: record)

        result = gtm_agent.get_prospect.invoke({"prospect_id": PROSPECT_ID})

        self.assertEqual(
            result,
            {
                "prospect": {
                    "prospect_id": PROSPECT_ID,
                    "name": "Taylor Prospect",
                    "email": "taylor@example.com",
                    "disqualified": False,
                    "annual_revenue": 5000000,
                    "enrichment_source": "CRM",
                },
                "found": True,
            },
        )

    def test_get_prospect_handles_absent_optional_fields(self):
        record = {
            "name": "Taylor Prospect",
            "email": "taylor@example.com",
            "billing_qualification": BILLING_QUALIFICATION,
            "unlisted_field": "not for the agent",
        }
        self.patch_data_service("get_prospect_record", lambda _: record)

        result = gtm_agent.get_prospect.invoke({"prospect_id": PROSPECT_ID})

        self.assertEqual(
            result,
            {
                "prospect": {
                    "prospect_id": PROSPECT_ID,
                    "name": "Taylor Prospect",
                    "email": "taylor@example.com",
                },
                "found": True,
            },
        )

    def test_build_prospect_profile_filters_data_from_result_and_cache(self):
        record = {
            "name": "Taylor Prospect",
            "email": "taylor@example.com",
            "annual_revenue": 5000000,
            "billing_qualification": BILLING_QUALIFICATION,
        }
        profiles = self.profile_cache()
        self.patch_data_service("get_prospect_record", lambda _: record)
        self.patch_data_service(
            "get_profile_from_db",
            lambda prospect_id: {"prospect_profile": profiles.get(prospect_id)},
        )
        self.patch_data_service(
            "save_profile_to_db",
            lambda prospect_id, profile: profiles.__setitem__(prospect_id, profile),
        )
        self.patch_data_service("fetch_engagement_history", lambda _: ["opened email"])
        self.patch_data_service("fetch_account_details", lambda _: {"industry": "software"})
        self.patch_data_service("fetch_tech_stack", lambda _: ["Python"])

        result = gtm_agent.build_prospect_profile.invoke({"prospect_id": PROSPECT_ID})
        profile = result["prospect_profile"]

        self.assertNotIn("billing_qualification", profile)
        self.assertNotIn("billing_qualification", profiles[PROSPECT_ID])
        self.assertEqual(profile["name"], "Taylor Prospect")
        self.assertEqual(profile["email"], "taylor@example.com")
        self.assertEqual(profile["annual_revenue"], 5000000)
        self.assertEqual(profile["engagement_history"], ["opened email"])
        self.assertEqual(profile["account_details"], {"industry": "software"})
        self.assertEqual(profile["tech_stack"], ["Python"])

        scoring_llm = Mock()
        scoring_llm.invoke.return_value.model_dump.return_value = {"score": 90}
        self.patch_data_service("fetch_tech_stack", lambda _: ["Python"])
        scoring_patch = patch.object(gtm_agent, "_scoring_llm", scoring_llm)
        scoring_patch.start()
        self.addCleanup(scoring_patch.stop)
        gtm_agent.score_prospect.invoke(
            {
                "prospect_profile": profile,
                "offering": {
                    "required_tech_stack": ["Python"],
                    "min_annual_revenue": 1000000,
                    "description": "A software offering",
                },
            }
        )
        scoring_input = scoring_llm.invoke.call_args.args[0][1]["content"]

        self.assertNotIn("billing_qualification", scoring_input)
        self.assertIn("Taylor Prospect", scoring_input)

    def test_build_prospect_profile_sanitizes_existing_cached_profile(self):
        profiles = self.profile_cache()
        profiles[PROSPECT_ID] = {
            "prospect_id": PROSPECT_ID,
            "name": "Taylor Prospect",
            "billing_qualification": BILLING_QUALIFICATION,
        }
        self.patch_data_service(
            "get_profile_from_db",
            lambda prospect_id: {"prospect_profile": profiles.get(prospect_id)},
        )
        self.patch_data_service(
            "save_profile_to_db",
            lambda prospect_id, profile: profiles.__setitem__(prospect_id, profile),
        )

        result = gtm_agent.build_prospect_profile.invoke({"prospect_id": PROSPECT_ID})

        self.assertNotIn("billing_qualification", result["prospect_profile"])
        self.assertNotIn("billing_qualification", profiles[PROSPECT_ID])


if __name__ == "__main__":
    unittest.main()
