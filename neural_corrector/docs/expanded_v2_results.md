# Expanded Neural-Only v2 Results

## Training

- Dataset: 1,068 examples (853 changed, 215 identity)
- Effective train examples: 774
- Validation examples: 103
- Synthetic examples: 0
- Seed: 1701
- Batch size: 4
- Early stopping: epoch 16
- Best validation loss: 0.820244
- Selected validation threshold: 0.90
- Model: `char-edit-bigru-0.2.0-expanded-neural-only`

One training row containing an unresolved `*` review placeholder was excluded
by the existing training guard.

## Locked Comparison

Both models were evaluated on the same expanded locked splits.

| Metric | v1 | v2 | Change |
|---|---:|---:|---:|
| Real-test correction F1 | 0.5330 | 0.7757 | +0.2426 |
| Real-test character error rate | 10.60% | 6.24% | -41.16% relative |
| Real-test word error rate | 55.48% | 29.40% | -47.00% relative |
| Real-test exact sentence match | 9.09% | 22.73% | +13.64 points |
| Clean sentence preservation | 51.43% | 91.43% | +40.00 points |
| Clean false corrections | 26 | 3 | -88.46% |
| Challenge correction F1 | 0.4627 | 0.6948 | +0.2321 |
| Challenge word error rate | 47.94% | 29.77% | -37.90% relative |
| Mean real-test inference | 13.25 ms | 12.19 ms | 8.00% faster |

## Spot Checks

| Input | v1 | v2 |
|---|---|---|
| `mort namel ikel` | `Mort namel ikel.` | `Mort nagħmel ikel.` |
| `illejla se nohrog` | `Illejla se noħrog` | `Illejla se noħroġ.` |
| `hafna snin ilu` | `hafna snin ilu` | `hafn snin ilu.` |
| `bongu` | `bongu` | `bongu` |

The expanded data produces a large measured improvement, but the remaining
spot-check failures confirm that this compact character tagger does not yet
generalize every learned orthographic pattern. Dictionary and suffix evidence
remain appropriate later controlled additions.

## Optional Dictionary Guard

An exact, tagged SQLite index was built from all 15 `.dic` sources:

- Source rows: 304,307
- Unique normalized forms: 206,069
- Index size: 6,156,288 bytes

The guard preserves a source token when the neural replacement is not a
dictionary-valid form. Case-only changes and valid article/preposition
components remain allowed.

| Metric | Neural-only v2 | v2 + dictionary guard |
|---|---:|---:|
| Real-test correction F1 | 0.7757 | 0.7694 |
| Real-test precision | 0.8361 | 0.8483 |
| Real-test word error rate | 29.40% | 29.24% |
| Clean sentence preservation | 91.43% | 97.14% |
| Clean false corrections | 3 | 1 |
| Challenge correction F1 | 0.6948 | 0.6905 |

For `tazez`, the raw neural output is `Tażeż`. The index recognizes `tazez`
as a noun and rejects the invented replacement, preserving `tazez`.
