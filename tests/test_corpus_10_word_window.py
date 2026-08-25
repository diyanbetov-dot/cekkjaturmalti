import unittest
from pathlib import Path

from app import CORPUS_RANKER, CORRECTOR, correct_text, normalize
from corpus_ranker import CorpusCandidateRanker, CorpusEvidence


class TestCorpus10WordWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ranker = CORPUS_RANKER

    def test_window_evidence_up_to_10_words(self):
        """Test window_evidence evaluates context up to 10 tokens on left and right."""
        left_ctx = ["Ġanni", "qabad", "l-ittra", "l-antika", "mingħand", "il-postier", "dalgħodu", "u", "skanta", "malli"]
        right_ctx = ["sew", "u", "fhem", "kollox", "li", "kien", "miktub", "dwar", "din", "il-ġrajja"]
        
        ev = self.ranker.window_evidence("qara", left_context=left_ctx, right_context=right_ctx, max_distance=10)
        self.assertIsInstance(ev, CorpusEvidence)
        self.assertGreater(ev.score, 0.0)

    def test_morphological_stem_unigram_fallback(self):
        """Test suffixed verb forms receive stem unigram fallback when unigram is missing."""
        ev_suffixed = self.ranker.window_evidence("qaraha", max_distance=10)
        ev_base = self.ranker.window_evidence("qara", max_distance=10)
        
        # Suffixed form should inherit stem unigram frequency credit
        self.assertGreater(ev_suffixed.unigram, 0.0)
        self.assertAlmostEqual(ev_suffixed.unigram, 0.75 * ev_base.unigram, delta=1e-4)

    def test_confusable_pair_xaghar_vs_xahar(self):
        """Test disambiguation between xagħar (hair) and xahar (month) in full 10-word contexts."""
        res_xahar, _ = correct_text("Kull xahar ikun hemm laqgħa importanti fl-uffiċċju tagħna.")
        self.assertIn("xahar", res_xahar)

        res_xaghar, _ = correct_text("Ir-raġel mar għand il-barbier u għandu rasu mimlija xagħar.")
        self.assertIn("xagħar", res_xaghar)

    def test_corpus_context_stops_at_sentence_and_quote_boundaries(self):
        corrected_period, _ = correct_text("Ara veru ta. BEEEEEP.")
        corrected_quote, tokens = correct_text('Ara veru ta" BEEEEEP.')

        self.assertEqual(corrected_period, "Ara veru ta. BEEEEEP.")
        self.assertEqual(corrected_quote, 'Ara veru ta" BEEEEEP.')
        ta_token = next(token for token in tokens if token.get("original") == "ta")
        self.assertNotEqual(ta_token.get("corrected"), "ta'")
        self.assertFalse(bool(ta_token.get("corpus_reordered")))

    def test_same_sentence_context_still_reranks_candidates(self):
        corrected, tokens = correct_text("kien amel hekk")
        amel = next(token for token in tokens if token.get("original") == "amel")

        self.assertEqual(corrected, "Kien għamel hekk.")
        self.assertTrue(bool(amel.get("corpus_reordered")))

    def test_verb_do_3sf_suffix_candidate_generation(self):
        """Test verb -a -> -ha candidate generation produces valid suffixed choices (e.g. aqra -> aqraha, semma -> semmieha)."""
        chosen, candidates, known = CORRECTOR.correct_word("aqra")
        self.assertTrue(known)
        cand_words = [c.word for c in candidates]
        self.assertIn("aqraha", cand_words)

        _, candidates_semma, _ = CORRECTOR.correct_word("semma")
        cand_words_semma = [c.word for c in candidates_semma]
        self.assertIn("semmieha", cand_words_semma)

    def test_apostrophe_preposition_preservation(self):
        """Test apostrophe-prefixed preposition compounds are preserved without distortion."""
        words = ["b'idejk", "m'ibni", "m'għandekx", "b'idea", "f'daqqa", "f'ġieħ", "t'ommok"]
        for w in words:
            self.assertTrue(CORRECTOR.is_known(w), f"Expected is_known({w}) to be True")
            corr, _ = correct_text(w)
            self.assertEqual(corr, w)

    def test_hyphenated_compound_preservation(self):
        """Test 131 hyphenated article/preposition compounds are recognized and preserved."""
        compounds = ["fl-ilma", "mill-ilma", "Taċ-ċirku", "Fiċ-ċentru", "biċ-ċavetta", "saċ-ċimiterju"]
        for comp in compounds:
            self.assertTrue(CORRECTOR.is_known(comp), f"Expected is_known({comp}) to be True")
            corr, _ = correct_text(comp)
            self.assertEqual(corr, comp)

    def test_all_caps_word_preservation(self):
        """Test ALL-CAPS words are recognized and not mis-corrected."""
        all_caps = ["JARAX", "BIEX", "FUQEK", "OFFRU", "MHUX", "DAN", "ĊAMA"]
        for w in all_caps:
            self.assertTrue(CORRECTOR.is_known(w), f"Expected is_known({w}) to be True")
            corr, _ = correct_text(w)
            self.assertEqual(corr, w)


    def test_homophone_phonological_alternative_injection(self):
        """Test homophone and phonological variants (xahar <-> xagħar, dehru <-> deru) are injected into candidates."""
        _, candidates_xahar, _ = CORRECTOR.correct_word("xahar")
        cand_words_xahar = [c.word for c in candidates_xahar]
        self.assertIn("xagħar", cand_words_xahar)

        _, candidates_dehru, _ = CORRECTOR.correct_word("dehru")
        cand_words_dehru = [c.word for c in candidates_dehru]
        self.assertIn("deru", cand_words_dehru)


    def test_nattanus_sentence_and_irnexilu_suggestions(self):
        """Test full correction of Nattanus sentence and clean irnexilu -> rnexxielu/rnexxilu suggestions."""
        inp = "Nattanus beda jigri. min qalb in nis kellu jahrab l ifirsa  addejin warajħ, ax ma setax jinqabad issa, mhux issa li irnexilu jgibu."
        exp = "Nattanus beda jiġri. Minn qalb in-nies kellu jaħrab l-ifirsa għaddejjin warajh, għax ma setax jinqabad issa, mhux issa li rnexxielu jġibu."
        
        corr, _ = correct_text(inp)
        self.assertEqual(corr, exp)

        chosen, candidates, _ = CORRECTOR.correct_word("irnexilu")
        cand_words = [c.word for c in candidates]
        self.assertIn("rnexxielu", cand_words)
        self.assertIn("rnexxilu", cand_words)


if __name__ == "__main__":
    unittest.main()
