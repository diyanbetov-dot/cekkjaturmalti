"""scratch/test_hybrid_branch.py

Test suite for the Hybrid-First Corrector (port 5002) and comparison with
the main pipeline (port 5000) and neural-only app (port 5001).

Run with:
    .\.venv\Scripts\python.exe scratch/test_hybrid_branch.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Force UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Bootstrap project root ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print("Loading spellchecker (this takes ~12-15s on first run)...")
from Essentials.core.spellchecker import UniversalMalteseSpellchecker, DICTIONARY_FILES, BASE_DIR
from Essentials.dictionary_meanings import MeaningIndex
from Essentials.helpers.article_phrase_rules import MalteseArticlePhraseRules
from Essentials.grammar import MalteseGrammarRuleEngine
from Essentials.helpers.suffix_generator import MalteseSuffixGenerator
from Essentials.helpers.orthographic_generator import MalteseOrthographicGenerator
from Essentials.helpers.composite_generator import MalteseCompositeGenerator
from Essentials.helpers.fused_preposition_rules import MalteseFusedPrepositionRules

spellchecker = UniversalMalteseSpellchecker(dictionary_files=DICTIONARY_FILES)
meaning_index = MeaningIndex()
meaning_index.load_entries(spellchecker.raw_entries, include_verbs=True)
spellchecker.raw_entries = []

article_phrase_rules = MalteseArticlePhraseRules(
    meaning_index=meaning_index,
    normalizer=spellchecker._normalize_word,
    noun_words=spellchecker.tagged_words_with_marker("NOUN"),
    num_words=spellchecker.tagged_words_with_marker("NUM"),
)
article_phrase_rules.spellchecker = spellchecker
spellchecker.article_phrase_rules = article_phrase_rules

grammar_rule_engine = MalteseGrammarRuleEngine(
    rules_path=BASE_DIR / "grammar" / "grammar_rules_measured.json",
    spellchecker=spellchecker,
    meaning_index=meaning_index,
    article_rules=article_phrase_rules,
)
spellchecker.grammar_rule_engine = grammar_rule_engine

suffix_generator = MalteseSuffixGenerator(
    spellchecker=spellchecker,
    verbs_file=[
        BASE_DIR / "finaldics/verbmt_semitic.dic",
        BASE_DIR / "finaldics/verbmt_nonsemitic.dic",
    ],
)
spellchecker.suffix_generator = suffix_generator
spellchecker.finalize_usage_verb_mappings()

orthographic_generator = MalteseOrthographicGenerator(spellchecker=spellchecker)
spellchecker.orthographic_generator = orthographic_generator
composite_generator = MalteseCompositeGenerator(spellchecker=spellchecker)
spellchecker.composite_generator = composite_generator
fused_preposition_rules = MalteseFusedPrepositionRules(
    spellchecker=spellchecker,
    article_rules=article_phrase_rules,
    meaning_index=meaning_index,
)
spellchecker.fused_preposition_rules = fused_preposition_rules
spellchecker.clear_disposable_startup_caches()

print("Loading neural corrector...")
from neural_corrector.inference.corrector import NeuralCorrector
ARTIFACT = ROOT / "neural_corrector" / "artifacts" / "char_edit_bigru_v4"
neural_corrector = NeuralCorrector(ARTIFACT)

print("Loading hybrid pipeline...")
from hybrid_corrector.pipeline import HybridFirstCorrector
pipeline = HybridFirstCorrector(
    spellchecker=spellchecker,
    neural_corrector=neural_corrector,
    grammar_rule_engine=grammar_rule_engine,
)

print("\n" + "=" * 70)
print("HYBRID-FIRST CORRECTOR — TEST SUITE")
print("=" * 70 + "\n")

# ── Test cases ──────────────────────────────────────────────────────────────
# Format: (input, expected_substring_in_output, description)
test_cases = [
    # Classic known-problem sentences
    ("ilbierah mort sal bahar biex nowm ftit.",
     "sal-baħar",
     "Preposition phrase: sal bahar → sal-baħar"),

    ("mort nghum ilbierah.",
     "ilbieraħ",
     "Adverb diacritic: ilbierah → ilbieraħ (must NOT become lbierah)"),

    ("ktibt ittra lill-kbira tieghu imma ma wegibtx.",
     "tiegħu",
     "Possessive diacritic: tieghu → tiegħu"),

    ("xtaqt niehu kafè mal habib tieghi.",
     "ħabib",
     "Diacritic in loanword context: habib → ħabib"),

    ("kellimna dak ir-ragel li kien jaf kollox.",
     "raġel",
     "Diacritic: ragel → raġel"),

    # Suffix/grammar tests
    ("rajt il-klieb jiġru fil-gnien.",
     "ġnien",
     "Diacritic: gnien → ġnien"),

    ("huwa jkellem lit-tfal b'mod sabih.",
     "b'mod",
     "Prepositional phrase: b'mod preserved"),

    # Neural should stay hands-off where main pipeline is confident
    ("il-qattus qieghed fuq il-mejda.",
     "qiegħed",
     "Diacritic: qieghed → qiegħed"),
]

passed = 0
failed = 0

for i, (inp, expected_substr, desc) in enumerate(test_cases, 1):
    result = pipeline.correct(inp, include_grammar=True)
    output = result["corrected_text"]
    stage1 = result["debug"]["stage1_text"]
    stage2 = result["debug"]["stage2_text"]
    ok = expected_substr in output
    status = "✓ PASS" if ok else "✗ FAIL"
    if ok:
        passed += 1
    else:
        failed += 1

    print(f"[{status}] Test {i}: {desc}")
    print(f"  Input:   {inp}")
    print(f"  Stage 1: {stage1}")
    print(f"  Stage 2: {stage2}")
    print(f"  Final:   {output}")
    if not ok:
        print(f"  EXPECTED to contain: '{expected_substr}'")
    print()

print("=" * 70)
print(f"Results: {passed}/{len(test_cases)} passed, {failed} failed")
print("=" * 70)

# ── Suggestion label check ──────────────────────────────────────────────────
print("\n── Suggestion label check ──────────────────────────────────────────")
print("Verifying no noisy neural labels appear in token choices...\n")

BAD_LABELS = {
    "Neural whole-word alternative",
    "The neural model predicts this spelling",
    "The neural model restored għ",
}

label_issues = 0
for inp, _, desc in test_cases:
    result = pipeline.correct(inp)
    for tok in result.get("tokens", []):
        if isinstance(tok, dict) and tok.get("type") == "word":
            for ch in tok.get("choices", []):
                if isinstance(ch, dict):
                    meaning = ch.get("meaning", "")
                    if meaning in BAD_LABELS:
                        print(f"  ✗ BAD LABEL in token for '{inp}': '{meaning}'")
                        label_issues += 1

if label_issues == 0:
    print("  ✓ No noisy neural labels found in any token choices.")
else:
    print(f"  ✗ {label_issues} bad labels found!")

print("\nDone.")
