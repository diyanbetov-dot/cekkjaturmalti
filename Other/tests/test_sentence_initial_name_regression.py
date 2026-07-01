import unittest

import app


class SentenceInitialNameRegressionTests(unittest.TestCase):
    def check_text(self, source):
        return app.spellchecker.correct_text_rich(
            source,
            edit_distance_tolerance=2,
        )

    def test_lowercase_sentence_initial_word_is_capitalized_without_name_like(self):
        result = self.check_text("ilbierah mort")
        self.assertEqual(result["corrected_text"], "Ilbieraħ mort.")
        self.assertFalse(result["tokens"][0]["name_like"])

    def test_unknown_sentence_initial_name_is_not_forced_into_random_correction(self):
        result = self.check_text("Diyan mar")
        self.assertEqual(result["corrected_text"], "Diyan mar.")
        self.assertFalse(result["tokens"][0]["name_like"])

    def test_second_unknown_sentence_initial_name_is_not_forced_into_random_correction(self):
        result = self.check_text("Basti mar")
        self.assertEqual(result["corrected_text"], "Basti mar.")
        self.assertFalse(result["tokens"][0]["name_like"])


if __name__ == "__main__":
    unittest.main()
