# Architecture Comparison

| Architecture | Strengths | Main risks | Baseline decision |
| --- | --- | --- | --- |
| Character edit tagger | Copies by default; handles diacritics, punctuation, spacing, insertion, deletion, replacement; small and local | One action per source character limits very long insertions; weak context with little data | Selected |
| Character encoder-decoder | Flexible sequence generation and word-boundary edits | Hallucination and clean-text damage; needs more data | Later comparison |
| Subword encoder-decoder | Better long-range context and efficient text length | Tokenization can obscure Maltese character errors; larger | Later comparison |
| BERTu plus correction head | Strong Maltese context; trainable project-specific head | External weights, GPU need, token-to-character alignment | Controlled experiment |
| BERTu reranker | Useful for ambiguity without generating text | Cannot recover candidates the generator missed | Later ablation |
| Custom model plus corpus | Can favour natural contextual candidates | Frequency can favour common but wrong forms | Later ablation |
| Hybrid runtime validator | Can suppress impossible morphology and names | Risks rebuilding the frozen rule engine | Only narrow measured use |

The first model is a two-layer bidirectional GRU over Unicode characters. At
each source character it predicts `<COPY>`, `<DELETE>`, or a literal output
string attached to that character. This supports replacements and local
insertions while retaining a strong copy path.

This architecture is the smallest meaningful learned corrector for the current
dataset. It establishes whether supervised neural evidence can provide useful
edits before external context or linguistic validators are introduced.

