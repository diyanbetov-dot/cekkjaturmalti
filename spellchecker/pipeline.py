import time
from typing import Dict, Any, List
from .tokenizer import tokenize_text
from .config import FINAL_DICS_DIR
from lexicon.indexes import LexiconIndexes
from lexicon.english import EnglishLexicon
from lexicon.entities import EntityLexicon
from candidates.generator import CandidateGenerator
from decode.decoder import GlobalDecoder
from decode.renderer import PureRenderer
from neural.runtime import NeuralRuntime


class SpellcheckerPipeline:
    def __init__(self) -> None:
        self.lexicon = LexiconIndexes(FINAL_DICS_DIR)
        self.english_lexicon = EnglishLexicon()
        self.entity_lexicon = EntityLexicon(self.lexicon.names_set)
        self.candidate_generator = CandidateGenerator(
            self.lexicon, self.english_lexicon, self.entity_lexicon
        )
        self.decoder = GlobalDecoder()
        self.renderer = PureRenderer()
        self.neural_runtime = NeuralRuntime()

    def check(self, text: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        tokens = tokenize_text(text)
        candidates = self.candidate_generator.generate_candidates(tokens)
        selected_edits = self.decoder.decode(candidates)
        corrected_text, edit_dicts = self.renderer.render(text, selected_edits)
        t1 = time.perf_counter()

        latency_ms = round((t1 - t0) * 1000, 2)
        return {
            "input_text": text,
            "corrected_text": corrected_text,
            "edits": edit_dicts,
            "suggestions": [],
            "latency_ms": latency_ms,
            "model": {
                "bertu_loaded": self.neural_runtime.loaded,
                "hybrid_heads_loaded": self.neural_runtime.loaded,
            },
        }
