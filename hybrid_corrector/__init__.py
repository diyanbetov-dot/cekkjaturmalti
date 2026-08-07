"""hybrid_corrector — Hybrid-First experimental spellchecker branch (port 5002).

Architecture:
  Stage 1 — Main pipeline (UniversalMalteseSpellchecker) for precision word fixes
  Stage 2 — Neural arbiter (NeuralCorrector) for context-sensitive recovery of
             unrecognised tokens only — locked words are never overridden
  Stage 3 — Grammar rule engine (same as main app)
"""
