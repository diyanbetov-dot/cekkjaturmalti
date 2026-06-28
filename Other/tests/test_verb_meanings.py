import unittest

import app
from dictionary_meanings import (
    format_verb_payload_meaning,
    parse_verb_payload,
)


class VerbMeaningFormattingTests(unittest.TestCase):
    def test_present_meanings_track_person(self):
        self.assertEqual(
            format_verb_payload_meaning("T-ghml-MPERF-1S-to make or do"),
            "I make or do",
        )
        self.assertEqual(
            format_verb_payload_meaning("T-ghml-MPERF-3SM-to make or do"),
            "he makes or does",
        )
        self.assertEqual(
            format_verb_payload_meaning("T-ghml-MPERF-3SF-to make or do"),
            "she makes or does",
        )
        self.assertEqual(
            format_verb_payload_meaning("T-ghml-MPERF-2P-to make or do"),
            "you all make or do",
        )

    def test_perfect_and_negative_meanings_track_tense(self):
        self.assertEqual(
            format_verb_payload_meaning("T-ghml-PERF-1S-to make or do"),
            "I made or did",
        )
        self.assertEqual(
            format_verb_payload_meaning("T-ghml-PERF-3P-to make or do"),
            "they made or did",
        )
        self.assertEqual(
            format_verb_payload_meaning("T-ghml-MPERF-3SM-to not make or not do-N"),
            "he doesn't make or do",
        )
        self.assertEqual(
            format_verb_payload_meaning("T-ghml-PERF-3SF-to not make or not do-N"),
            "she didn't make or do",
        )

    def test_imperative_meanings_are_commands(self):
        self.assertEqual(
            format_verb_payload_meaning("T-ghml-IMP-2S-to make or do"),
            "you, make or do (command)",
        )
        self.assertEqual(
            format_verb_payload_meaning("T-ghml-IMP-2P-to not make or not do-N"),
            "you all, don't make or do (command)",
        )

    def test_payload_parser_detects_negative_flag(self):
        parsed = parse_verb_payload("T-ksr-PERF-3SM-to not break or not divert-N")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["tense"], "PERF")
        self.assertEqual(parsed["person"], "3SM")
        self.assertEqual(parsed["gloss"], "to not break or not divert")
        self.assertTrue(parsed["negative"])


class VerbSuffixMeaningIntegrationTests(unittest.TestCase):
    def test_direct_object_suffix_meanings_are_expanded(self):
        self.assertEqual(
            app.spellchecker.meaning_for("kisruh"),
            "they broke or diverted him",
        )

    def test_indirect_object_suffix_meanings_are_possessive(self):
        self.assertEqual(
            app.spellchecker.meaning_for("kisrulu"),
            "they broke or diverted his",
        )
        self.assertEqual(
            app.spellchecker.meaning_for("kisrulha"),
            "they broke or diverted her",
        )
        self.assertEqual(
            app.spellchecker.meaning_for("kisrulhom"),
            "they broke or diverted their",
        )


class ExactVerbMeaningLookupTests(unittest.TestCase):
    def test_exact_semitic_verbs_have_meanings(self):
        self.assertEqual(
            app.spellchecker.meaning_for("kiteb"),
            "he wrote or registered",
        )
        self.assertEqual(
            app.spellchecker.meaning_for("kitibx"),
            "he didn't write or register",
        )
        self.assertEqual(
            app.spellchecker.meaning_for("jikteb"),
            "he writes or registers",
        )
        self.assertEqual(
            app.spellchecker.meaning_for("kitbu"),
            "they wrote or registered",
        )

    def test_exact_non_semitic_verbs_have_meanings(self):
        self.assertEqual(app.spellchecker.meaning_for("introduċa"), "he introduced")
        self.assertEqual(app.spellchecker.meaning_for("introduċew"), "they introduced")


class VerbMeaningEndpointTests(unittest.TestCase):
    def test_check_text_exposes_token_meaning_for_exact_verb(self):
        client = app.app.test_client()
        response = client.post(
            "/check-text",
            json={"text": "kiteb", "edit_distance_tolerance": 1},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["corrected_text"], "kiteb")
        self.assertEqual(len(payload["tokens"]), 1)
        self.assertEqual(payload["tokens"][0]["corrected"], "kiteb")
        self.assertEqual(
            payload["tokens"][0]["meaning"],
            "he wrote or registered",
        )


if __name__ == "__main__":
    unittest.main()
