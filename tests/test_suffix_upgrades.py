from pathlib import Path
import unittest

from app import CORRECTOR


class TestSuffixUpgrades(unittest.TestCase):
    def test_baghatilna_prioritizes_baghtilna(self):
        engine = CORRECTOR.suffix_engine
        self.assertIsNotNone(engine)
        suggestions = engine.suggestions("baghatilna")
        self.assertTrue(len(suggestions) > 0)
        self.assertEqual(suggestions[0], "bagħtilna")
        self.assertNotIn("bagħatlna", suggestions)
        if "bagħtulna" in suggestions:
            self.assertLess(
                suggestions.index("bagħtilna"),
                suggestions.index("bagħtulna"),
            )

    def test_alaqhulu_keeps_only_valid_skeleton_matches(self):
        engine = CORRECTOR.suffix_engine
        self.assertIsNotNone(engine)
        suggestions = engine.suggestions("alaqhulu")
        self.assertEqual(suggestions, ["għalaqhulu", "agħlaqhulu"])


if __name__ == "__main__":
    unittest.main()
