# Grammar Implementation Audit

Date: 2026-07-13

## Current startup path

- The repository root `app.py` is the process entry point used by local runs and by the current container command.
- `app.py` delegates to `Essentials.app`:
  - `from Essentials.app import *`
  - `from Essentials.app import app as flask_app`
- `gunicorn.conf.py` exists, but the current `Dockerfile` does not use it.
- The current `Dockerfile` starts `gunicorn` with `app:app`, `1` worker, `8` threads, and `timeout 0`.

## Current production and offline split

- There is no live grammar engine in the production path yet.
- The offline BERTu/UD kit is provided as standalone files in the workspace attachment, but nothing in the current repo imports `torch`, `transformers`, or `conllu` in production.
- `requirements.txt` currently contains only:
  - `Flask==3.1.3`
  - `gunicorn==23.0.0`

## Important data structures already in use

- The spellchecker returns a structured JSON response from `correct_text_rich`.
- The live response already contains:
  - `corrected_text`
  - `tokens`
- Token entries already carry fields such as:
  - `type`
  - `original`
  - `corrected`
  - `meaning`
  - `ambiguous`
  - `crucial`
  - `choices`
  - `name_like`
  - `unrecognized`
- This means a grammar layer should be additive to the current token model, not a separate front-end system.

## Current limits and pressure points

- The live checker already has a large dictionary/morphology footprint:
  - `166291` dictionary words
  - `3757` paradigms
  - `203977` suffix-verb records
- The current app is memory-heavy before any grammar work.
- The live path has bounded caches in many places, but the production branch should still be reviewed cache-by-cache before adding any more load.
- The current performance probe shows the app is already sensitive to expensive candidate generation on difficult words such as `zghazugha`.

## Likely memory pressure

- `Other/tools/memory_profile.py` reports:
  - `tracemalloc_current_mb`: about `345.56 MB`
  - `tracemalloc_peak_mb`: about `357.31 MB`
- On this Windows environment the RSS helper returned `null`, so true RSS was not measurable through that script here.
- The result still suggests the current live checker is already carrying a substantial in-memory index before any grammar layer is added.

## Likely latency pressure

Measured with `Other/tests/test_check_text_performance.py`:

- `required_phrase`: first request about `0.649 s`, warm median about `0.522 s`
- `difficult_word`: first request about `0.509 s`, warm median about `0.516 s`
- `four_easy_misspellings`: warm median about `0.472 s`
- `four_rule_families`: first request about `2.761 s`, warm median about `0.622 s`
- `mixed_social`: first request about `0.839 s`, warm median about `0.723 s`
- Two concurrent sample requests completed in about `0.50 s` and `1.14 s`

The main latency spike remains in hard candidate-generation cases, not in the baseline request path.

## Current regression state

Baseline regression check:

- `Other/tests/test_check_text_baseline.py` is currently failing on `memory_phrase`.
- The observed change is sentence-initial article capitalization:
  - expected: `il-bajja għalqet għax sabu`
  - actual: `Il-bajja għalqet għax sabu`
- This is a real existing regression against the current baseline fixture and should be treated before grammar work is layered on top.

Current contextual regression tests:

- `Other/tests/test_contextual_i_and_lexical_coverage.py` passes.

Current performance test:

- `Other/tests/test_check_text_performance.py` passes when `PYTHONIOENCODING=utf-8` is set.
- Without UTF-8 output on this Windows shell, the script can fail on Maltese characters while printing JSON.

## Integration point selected

The best integration point is after the current spelling and morphology resolution inside `Essentials.app.correct_text_rich`, using the existing token JSON rather than inventing a second response format.

Why:

- offsets are already computed there;
- spelling candidates and meanings already exist there;
- the front end already consumes that shape;
- a grammar layer can be made additive and reversible.

## Risks discovered

- The current baseline already has a quality regression unrelated to grammar.
- The app is already large enough that any grammar layer must be bounded carefully.
- The current Docker command ignores `gunicorn.conf.py`, so configuration drift is already present.
- The offline grammar kit is useful as a research source, but it should stay out of production dependencies.

## Recommendations before implementation

- Fix or intentionally re-baseline the current sentence-initial article capitalization regression before adding grammar features.
- Keep grammar strictly feature-flagged and off by default.
- Reuse the current token response schema and avoid a second highlighting system.
- Keep BERTu and UD tooling offline only.
- Replace the Docker/Gunicorn duplication with a single source of truth before benchmarking concurrency.
- Measure cache and memory behaviour again after any new layer is added.

## Deviations from the attachment that may be necessary later

- The repository currently already contains a substantial spellchecker with rich token output; a separate generic grammar engine would be redundant unless it is heavily constrained.
- The attachment assumes a new grammar layer can be added without first resolving existing baseline drift. The current audit shows that baseline drift already exists and should be addressed first.

