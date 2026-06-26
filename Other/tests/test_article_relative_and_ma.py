import unittest

import app


class ArticleRelativeAndMaTests(unittest.TestCase):
    def check_text(self, source, expected):
        result = app.spellchecker.correct_text_rich(
            source,
            edit_distance_tolerance=2,
        )
        self.assertEqual(result["corrected_text"], expected)
        return result

    def test_superlative_adjective_keeps_short_article(self):
        result = self.check_text("l-ikbar", "l-ikbar")
        self.assertEqual(
            app.spellchecker.meaning_for("l-ikbar"),
            "the largest, biggest, oldest",
        )
        self.assertIn(
            "l'ikbar",
            [choice["word"] for choice in result["tokens"][0]["choices"]],
        )

    def test_relative_l_apostrophe_is_not_an_article(self):
        self.check_text("l'ikbar", "l'ikbar")
        self.assertEqual(
            app.spellchecker.meaning_for("l'ikbar"),
            "which is larger, bigger, older",
        )
        self.check_text("l'eżistiet", "l'eżistiet")
        self.assertEqual(
            app.spellchecker.meaning_for("l'eżistiet"),
            "which existed",
        )

    def test_article_l_does_not_go_before_hawn_or_hemm(self):
        self.check_text("l-hawn", "hawn")
        self.check_text("l-hemm", "hemm")
        self.check_text("l'hawn", "l'hawn")
        self.check_text("l'hemm", "l'hemm")

    def test_negative_ma_contracts_before_a_vowel(self):
        self.check_text("ma eżistiet", "m'eżistiet")
        self.check_text("mezistiet", "m'eżistiet")
        self.check_text("ma marret", "ma marret")

    def test_mlt_tags_supply_meanings(self):
        self.assertEqual(app.meaning_index.meaning_for("Ċina"), "China")
        self.assertIn(
            "someone from Afghanistan",
            app.meaning_index.meanings_for("Afgan"),
        )


if __name__ == "__main__":
    unittest.main()
