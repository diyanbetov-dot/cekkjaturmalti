# Neural-First Maltese Corrector

This directory is an independent experiment. It does not import or modify the
frozen hybrid spellchecker's correction pipeline. The first baseline is a
trainable bidirectional-GRU character edit tagger learned from noisy-to-clean
Maltese pairs.

## Commands

```powershell
.\.venv\Scripts\python.exe -m neural_corrector.dataset.merge_examples --incoming <path> --identity-count 200 --apply
.\.venv\Scripts\python.exe -m neural_corrector.dataset.parse_ai_corrections
.\.venv\Scripts\python.exe -m neural_corrector.dataset.analyze_pairs
.\.venv\Scripts\python.exe -m neural_corrector.dataset.split_groups --force
.\.venv\Scripts\python.exe -m neural_corrector.dataset.audit_expanded_dataset
.\.venv\Scripts\python.exe -m neural_corrector.dataset.build_dictionary_index
.\.venv\Scripts\python.exe -m neural_corrector.training.train
.\.venv\Scripts\python.exe -m neural_corrector.evaluation.evaluate
.\.venv\Scripts\python.exe -m neural_corrector.correct "mort il bahar" --json
.\.venv\Scripts\python.exe -m neural_corrector.web.app --port 5001
```

The web command serves the existing UI unchanged at
`http://127.0.0.1:5001/`, backed only by the experimental neural corrector.

Measured baseline results are in `docs/baseline_results.md`. Expanded neural-only
results are in `docs/expanded_v2_results.md`; full machine reports are in
`experiments/`.

The expanded dataset currently contains 1,068 examples: 853 changed pairs and
215 identity pairs. Its locked grouped splits and leakage report are in
`data/splits/` and `data/reports/expanded_dataset_audit.json`.

## Baseline Boundary

The following optional components are disabled in
`configs/baseline.json`: BERTu, corpus evidence, dictionary validation,
morphology, suffix resources, preprocessor candidates, and postprocessing.
They may be enabled only through later measured ablations.

Synthetic training is also disabled. The existing `synthetic_train.jsonl`
belongs to the frozen first baseline and is retained only for provenance until
new dictionary- and suffix-labelled datasets are generated from the expanded
training partition.

The v2 inference configuration enables an exact dictionary-validation guard
backed by a compact read-only SQLite index. It may be disabled independently
without retraining by setting `use_dictionary_validation` to `false` in the
artifact's `inference_config.json`.

The model always returns structured edits. Multiple alternatives are exposed
through the current suggestion UI only for a single bare-word input. Text with
context receives the model's selected correction without that ambiguity UI
route in this first experiment.
