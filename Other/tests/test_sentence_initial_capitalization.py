import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import app


class SentenceInitialCapitalizationTests(unittest.TestCase):
    def check_text(self, source, expected):
        result = app.spellchecker.correct_text_rich(
            source,
            edit_distance_tolerance=2,
        )
        self.assertEqual(result["corrected_text"], expected)
        return result

    def test_first_word_is_capitalized_after_correction(self):
        self.check_text("ilbierah mort", "Ilbieraħ mort.")

    def test_word_after_question_mark_is_capitalized(self):
        self.check_text("mort? ilbierah mort", "Mort? Ilbieraħ mort.")

    def test_word_after_exclamation_mark_is_capitalized(self):
        self.check_text("mort! ilbierah mort", "Mort! Ilbieraħ mort.")

    def test_word_after_period_is_capitalized(self):
        self.check_text("mort. ilbierah mort", "Mort. Ilbieraħ mort.")

    def test_initial_article_phrase_is_capitalized_and_terminated(self):
        self.check_text("l bajja", "L-bajja.")


if __name__ == "__main__":
    unittest.main()
