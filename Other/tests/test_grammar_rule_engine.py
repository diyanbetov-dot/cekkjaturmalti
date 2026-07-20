# -*- coding: utf-8 -*-
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import app


class GrammarRuleEngineTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()

    def test_check_text_exposes_grammar_by_default(self):
        response = self.client.post(
            "/check-text",
            json={"text": "Jien min Malta.", "edit_distance_tolerance": 1},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["grammar_enabled"])
        self.assertIn("grammar_findings", payload)

    def test_check_text_can_expose_grammar_findings(self):
        response = self.client.post(
            "/check-text",
            json={
                "text": "Jien min Malta.",
                "edit_distance_tolerance": 1,
                "include_grammar": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["grammar_enabled"])
        rule_ids = {finding["rule_id"] for finding in payload["grammar_findings"]}
        self.assertIn("MIN_MINN_CONTEXT", rule_ids)

    def test_adjective_before_noun_is_flagged(self):
        response = self.client.post(
            "/check-text",
            json={
                "text": "Rajna ħamra karozza.",
                "edit_distance_tolerance": 1,
                "include_grammar": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        rule_ids = {finding["rule_id"] for finding in payload["grammar_findings"]}
        self.assertIn("AMOD_ORDER_NOUN_BEFORE_ADJ", rule_ids)
        finding = next(
            finding
            for finding in payload["grammar_findings"]
            if finding["rule_id"] == "AMOD_ORDER_NOUN_BEFORE_ADJ"
        )
        self.assertEqual(finding["suggestion"], "karozza ħamra")

    def test_numeral_noun_agreement_is_flagged(self):
        response = self.client.post(
            "/check-text",
            json={
                "text": "Qrajt ħames bozza u għoxrin bozza.",
                "edit_distance_tolerance": 1,
                "include_grammar": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        rule_ids = [finding["rule_id"] for finding in payload["grammar_findings"]]
        self.assertIn("NUMERAL_NOUN_NUMBER", rule_ids)
        number_findings = [
            finding
            for finding in payload["grammar_findings"]
            if finding["rule_id"] == "NUMERAL_NOUN_NUMBER"
        ]
        surfaces = {finding["surface"] for finding in number_findings}
        self.assertIn("ħames bozza", surfaces)
        self.assertIn("għoxrin bozza", surfaces)

    def test_number_correction_drives_plural_noun_surface(self):
        response = self.client.post(
            "/check-text",
            json={
                "text": "erba bozza.",
                "edit_distance_tolerance": 1,
                "include_grammar": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["corrected_text"], "Erba' bozoz.")

    def test_adjective_gender_agreement_is_flagged(self):
        response = self.client.post(
            "/check-text",
            json={
                "text": "karozza aħmar tinsab barra.",
                "edit_distance_tolerance": 1,
                "include_grammar": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        finding = next(
            finding
            for finding in payload["grammar_findings"]
            if finding["rule_id"] == "AMOD_GENDER_AGREEMENT"
        )
        self.assertEqual(finding["surface"], "karozza aħmar")
        self.assertTrue(finding.get("suggestion"))

    def test_adjective_number_agreement_is_flagged(self):
        response = self.client.post(
            "/check-text",
            json={
                "text": "karozzi kbira ġew.",
                "edit_distance_tolerance": 1,
                "include_grammar": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        finding = next(
            finding
            for finding in payload["grammar_findings"]
            if finding["rule_id"] == "AMOD_NUMBER_AGREEMENT"
        )
        self.assertEqual(finding["surface"], "karozzi kbira")
        self.assertTrue(finding.get("suggestion"))

    def test_subject_verb_person_number_is_flagged(self):
        response = self.client.post(
            "/check-text",
            json={
                "text": "mara marret jixtri.",
                "edit_distance_tolerance": 1,
                "include_grammar": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        finding = next(
            finding
            for finding in payload["grammar_findings"]
            if finding["rule_id"] == "SUBJECT_VERB_PERSON_NUMBER"
        )
        self.assertIn("mara marret", finding["surface"])
        self.assertTrue(finding.get("suggestion"))

    def test_subject_and_verb_chain_agreement_are_rewritten(self):
        cases = {
            "mara jixtri": "Mara tixtri.",
            "bniet jixtri": "Bniet jixtru.",
            "bniet marret": "Bniet marru.",
            "il mara marret jixtri": "Il-mara marret tixtri.",
        }
        for original, expected in cases.items():
            with self.subTest(original=original):
                response = self.client.post("/check-text", json={"text": original})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json()["corrected_text"], expected)

    def test_split_time_expressions_are_compounded(self):
        cases = {
            "il bierah": "Ilbieraħ.",
            "il lum": "Illum.",
            "il lejla": "Il-lejla.",
        }
        for original, expected in cases.items():
            with self.subTest(original=original):
                response = self.client.post("/check-text", json={"text": original})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json()["corrected_text"], expected)

    def test_definite_np_article_propagation_is_flagged(self):
        response = self.client.post(
            "/check-text",
            json={
                "text": "il karozza ħamra tinsab barra.",
                "edit_distance_tolerance": 1,
                "include_grammar": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        finding = next(
            finding
            for finding in payload["grammar_findings"]
            if finding["rule_id"] == "DEF_NP_ARTICLE_PROPAGATION"
        )
        self.assertIn("karozza ħamra", finding["surface"])
        self.assertTrue(finding.get("suggestion"))

    def test_contextual_ta_names_and_xi_contraction(self):
        cases = {
            "ta hanzir lil john": "Ta ħanżir lil John.",
            "Ma nafx xint tejd ta": "Ma nafx x'int tgħid ta.",
            "Insomma ta forma ta hanzir andu": (
                "Insomma ta forma ta' ħanżir għandu."
            ),
            "guh ta hanzir": "Ġuħ ta' ħanżir.",
            "dan il basket ta min hu": "Dan il-basket ta' min hu?",
            "lbierah ta daqqa ta ponn lil john": (
                "Ilbieraħ ta daqqa ta' ponn lil John."
            ),
        }

        for original, expected in cases.items():
            with self.subTest(original=original):
                response = self.client.post(
                    "/check-text",
                    json={
                        "text": original,
                        "edit_distance_tolerance": 1,
                        "include_grammar": True,
                    },
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json()["corrected_text"], expected)


    def test_ambiguous_verb_noun_prefers_noun_adjective_agreement(self):
        response = self.client.post(
            "/check-text",
            json={
                "text": "triq twil",
                "edit_distance_tolerance": 2,
                "include_grammar": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["corrected_text"], "Triq twila.")
        rule_ids = {finding["rule_id"] for finding in payload["grammar_findings"]}
        self.assertIn("AMOD_GENDER_AGREEMENT", rule_ids)
        self.assertNotIn("VERB_ADJECTIVE_COMPATIBILITY", rule_ids)

    def test_contextual_initial_i_for_imperfect_verbs(self):
        cases = {
            "Mort ndur dawra mal belt": "Mort indur dawra mal-belt.",
            "Morna induru dawra mal belt": "Morna nduru dawra mal-belt.",
            "wara inmut": "Wara mmut.",
        }
        for original, expected in cases.items():
            with self.subTest(original=original):
                response = self.client.post(
                    "/check-text",
                    json={"text": original, "edit_distance_tolerance": 2},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json()["corrected_text"], expected)

    def test_verb_adjective_compatibility_rule(self):
        invalid = self.client.post(
            "/check-text",
            json={
                "text": "Jitwal e\u0127xen",
                "edit_distance_tolerance": 2,
                "include_grammar": True,
            },
        ).get_json()
        finding = next(
            finding
            for finding in invalid["grammar_findings"]
            if finding["rule_id"] == "VERB_ADJECTIVE_COMPATIBILITY"
        )
        self.assertEqual(finding["suggestion"], "Jitwal u jsir e\u0127xen")

        accepted_cases = [
            "Jikber e\u0127xen",
            "Jsir e\u0127xen",
            "Jdum itwal",
            "Idum iqsar",
            "Jitwieled kbir",
        ]
        for original in accepted_cases:
            with self.subTest(original=original):
                response = self.client.post(
                    "/check-text",
                    json={
                        "text": original,
                        "edit_distance_tolerance": 2,
                        "include_grammar": True,
                    },
                )
                self.assertEqual(response.status_code, 200)
                rule_ids = {
                    finding["rule_id"]
                    for finding in response.get_json()["grammar_findings"]
                }
                self.assertNotIn("VERB_ADJECTIVE_COMPATIBILITY", rule_ids)


if __name__ == "__main__":
    unittest.main()
