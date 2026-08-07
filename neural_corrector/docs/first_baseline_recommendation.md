# First Baseline Recommendation

Train and evaluate the custom character edit tagger before adding any old
subsystem.

Reasons:

1. It learns trainable correction parameters directly from project examples.
2. Its copy action gives an explicit path to preserving clean text.
3. It supports Maltese diacritics, apostrophes, hyphens, capitalization,
   punctuation, merges, splits, and local morphology in one objective.
4. It is feasible on the current CPU-only PyTorch installation.
5. It produces per-position probabilities that can become structured edits,
   confidence, categories, and alternatives.
6. Its small size makes later BERTu/corpus/dictionary improvements measurable
   instead of hiding them inside a large initial system.

The key limitation is data volume. Results should be treated as an honest
baseline, not production quality. Threshold selection must use validation data
only and should prioritize correction precision and clean-text preservation.

The first UI adapter exposes alternative suggestions only for an isolated bare
word. In contextual text the neural model still emits structured edits, but the
adapter displays the selected correction without a multi-choice ambiguity box.
This prevents an uncalibrated alternative generator from flooding normal text.

