# Commit 3 broader-suite baseline

Command run before completing Commit 3:

```text
.venv\Scripts\python.exe -m pytest neural_corrector\tests -q --tb=short
```

Result: **7 failed, 10 passed in 131.13s**.

These failures pre-date Commit 3. Commit 3 changes only `contextual_corrector`; it does
not modify the neural dataset, locked splits, inference behavior, or period handling.

## Stale dataset counts and splits

1. `test_parser_keeps_every_documented_block`
   Expected `1068`; the current parsed dataset contains `1368` blocks.
2. `test_incoming_merge_is_recoverable_and_idempotent`
   Expected `1068`; the rebuilt merge report contains `1370` examples.
3. `test_locked_splits_are_disjoint_and_complete`
   The current dataset has IDs `ai-corrections-001364` through
   `ai-corrections-001368` that are absent from the locked split files.
4. `test_normalized_groups_do_not_cross_dataset_partitions`
   Split lookup raises `KeyError: 'ai-corrections-001364'` for the same stale-split
   condition.

## Stale period expectations

5. `test_dictionary_rescue_recovers_high_confidence_valid_word`
   Expected `Għamilt erba tazez`; current output is `Għamilt erba tazez.`.
6. `test_plural_noun_selects_short_attributive_number_candidate`
   Expected `erba' tazez`; current output is `erba' tazez.`.
7. `test_low_confidence_prefix_candidate_is_not_rescued`
   Expected `Mort nagħmel erba' tazez`; current output is
   `Mort nagħmel erba' tazez.`.

The failures are recorded, not repaired, because changing those unrelated tests or
artifacts is outside Commit 3.
