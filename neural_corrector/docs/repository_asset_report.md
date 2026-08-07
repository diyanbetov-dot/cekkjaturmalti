# Repository Asset Report

## Safety Snapshot

- Frozen `main` commit: `af422ec34cb8ea4d1fa7fc9d18414101ce9eb8bd`.
- Experimental branch: `codex/neural-first-experiment`.
- Pre-experiment checkpoint: `9113477fdc4a2411ab7a49a2b2c924e8286e3c46`.
- All new implementation files live under `neural_corrector/`.
- The existing app, UI, dictionaries, helpers, and tests are read-only sources.

## Supervised Correction Data

`AI corrections.txt` is the primary manual noisy-to-clean source. Its SHA-256
is `47e89d3d7d49bb9cc9ed1a340ae09ab631be8b128ac09fa2568ff7535858c575`.
The parser found 212 blocks, retained all 212, reported zero malformed blocks,
15 unchanged examples, 19 multiline examples, 211 unique inputs, 194 unique
outputs, and no exact duplicate pairs or conflicting duplicate inputs.

Four outputs are flagged rather than silently changed. Two contain apparent
visual line-wrap hyphens; two contain review placeholders for `xar` and
`ntbat`. The placeholder examples remain in the processed dataset but are
excluded from training and synthetic generation.

The repository also contains extensive hybrid regression tests under
`Other/tests/`. Useful families cover baseline output, performance, corpus,
candidate evidence, contractions, contextual `i-`, English handling, names,
number-noun agreement, suffixes, verb meanings, long text, and recorded social
comments. These are independent evaluation sources until explicitly approved
for training.

## Dictionaries

`Essentials/finaldics/` contains lexical and tagged resources:

| Resource | Approximate size | Potential neural use |
| --- | ---: | --- |
| `verbmt_semitic.dic` | 8.0 MB | morphology coverage, synthetic examples, validation |
| `verbmt_nonsemitic.dic` | 3.7 MB | borrowing/verb coverage and morphology |
| `names.dic` | 530 KB | optional name protection and evaluation |
| `places.dic` | 474 KB | optional place handling |
| `fixednouns.dic` | 409 KB | noun coverage and meaning metadata |
| `maltese_adjectives.dic` | 130 KB | agreement/morphology evaluation |
| Other `.dic` files | under 100 KB each | articles, pronouns, prepositions, usage, participles |

Formats are line-oriented and often include slash-delimited tags or meanings.
They require dedicated parsers before feature or validation use. The baseline
does not treat them as correction authorities.

## Corpus And Context Assets

The local corpus metadata identifies a Korpus Malti v4.2 MLRS subset with
1,240,023 valid tokens, 64,001 sentences, and a 32,721-item vocabulary.
Existing indexes total roughly 1.4 MB. The metadata also records 205,174
malformed rows, so provenance and filtering matter. Existing helpers include
`corpus_scorer.py` and `corpus_context_selector.py`.

Corpus evidence is excluded from baseline inference. Later controlled uses are
clean-text extraction, synthetic corruption, coverage analysis, and candidate
reranking. Frequency alone must never mean correctness.

## Morphology, Suffixes, And BERTu

- `suffix_generator.py`, `suffix_rules.py`, and `verb_form_index.py` contain
  valuable verb and attached-pronoun knowledge.
- Their preferred first use is synthetic-data generation and challenge-set
  construction, not runtime rewriting.
- `bertu_reranker.py` is an existing optional contextual integration.
- BERTu may later act as a frozen encoder, trainable backbone, reranker,
  ambiguity resolver, teacher, or confidence estimator.
- None of these components participates in baseline-one inference.

## Data Quality Risks

- Only 212 manual examples are available; this is very small for a general
  sequence corrector.
- Examples range from 3 to 1,471 input characters, with a median of 10.
- Long social-text examples can dominate character-level training unless
  chunked and sampled carefully.
- Fifteen unchanged examples are insufficient to guarantee strong
  preservation without additional clean training data.
- Some expected outputs are debatable or explicitly marked for review.
- Near-related paradigms create leakage risk; locked group-aware splits are
  therefore required.
- Rare characters, apostrophe variants, emoji, and possible mojibake are
  reported rather than normalized away.

## Licensing And Redistribution

- `AI corrections.txt` is a project-maintained source, but any personal or
  third-party text within it should be reviewed before public redistribution.
- Korpus Malti/MLRS material must retain its original attribution and license;
  raw corpus redistribution is not assumed to be permitted.
- Dictionary origins include project notes referencing verb.mt, Wiktionary,
  Ġabra, and Kelmet il-Malti 2017. Each source's redistribution and derivative
  model terms need confirmation before publishing copied resources or weights.
- BERTu model and tokenizer licenses must be recorded before committing or
  distributing external weights.
- Large external weights and raw corpora should not be committed to Git.

## Hardware And Storage

The workstation has an RTX 3060 Laptop GPU with 6 GB VRAM, but the active
PyTorch build is CPU-only (`torch 2.13.0+cpu`). The current BiGRU baseline is
small enough to train on CPU in minutes and should occupy only a few megabytes.
Larger encoder-decoder or BERTu fine-tuning experiments require a CUDA-enabled
PyTorch environment and tighter memory management. The initial baseline does
not require downloading external models.

## Reuse Decision

Reuse now: `AI corrections.txt`, its locked splits, and synthetic corruptions
derived only from training outputs.

Keep disabled initially: hybrid rules, dictionaries, corpus scoring, BERTu,
suffix generation, morphology validation, and deterministic lexical fixes.

Evaluate later through ablations: clean corpus augmentation, dictionary
features, suffix-derived training data, BERTu context, and corpus reranking.

