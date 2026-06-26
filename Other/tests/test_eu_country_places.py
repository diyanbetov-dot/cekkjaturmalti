import unittest

import app


class EuCountryPlaceTests(unittest.TestCase):
    def test_official_meanings_are_loaded(self):
        self.assertEqual(app.meaning_index.meaning_for("Ċina"), "China")
        self.assertIn(
            "someone from Afghanistan",
            app.meaning_index.meanings_for("Afgan"),
        )

    def test_english_country_gets_one_way_maltese_choice(self):
        result = app.spellchecker.correct_text_rich("China")
        token = next(token for token in result["tokens"] if token["type"] != "text")
        self.assertEqual(token["corrected"], "China")
        self.assertTrue(token["place_translation"])
        self.assertEqual(token["choices"][0]["word"], "Ċina")
        self.assertEqual(token["choices"][0]["meaning"], "China")

    def test_maltese_country_does_not_offer_english(self):
        result = app.spellchecker.correct_text_rich("Ċina")
        token = next(token for token in result["tokens"] if token["type"] != "text")
        self.assertEqual(token["corrected"], "Ċina")
        self.assertFalse(token.get("place_translation", False))

    def test_english_country_typo_is_not_corrected_to_english(self):
        result = app.spellchecker.correct_text_rich("Chna")
        token = next(token for token in result["tokens"] if token["type"] != "text")
        self.assertNotEqual(token["corrected"], "China")
        self.assertFalse(token.get("place_translation", False))

    def test_country_does_not_absorb_following_conjunction(self):
        result = app.spellchecker.correct_text_rich("China u Ċina")
        self.assertEqual(result["corrected_text"], "China u Ċina")
        self.assertEqual(result["tokens"][0]["choices"][0]["word"], "Ċina")


if __name__ == "__main__":
    unittest.main()
