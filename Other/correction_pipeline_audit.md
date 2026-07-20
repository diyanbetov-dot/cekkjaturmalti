# Correction Pipeline Audit

Date: 2026-07-16

## Purpose

This note records the live correction order so future work does not add a
second rule for work that an earlier or later stage already owns.

## Current pipeline

1. `correct_text_rich` tokenises input and classifies protected spans.
   - Exact approved English stays English.
   - Exact approved English phrases stay together.
   - Quoted, unapproved text loses its quotes, remains unchanged, and is
     explicitly unrecognized.
2. Phrase rules consume multi-token Maltese structures such as articles,
   prepositions, negative forms, numerals, and fixed expressions.
3. `correct_word` performs lexical and morphology-driven repair for a single
   word.
4. `_phase_z_finalize_surface_word` selects a context-sensitive surface form,
   including case, terminal punctuation, and initial-vowel variants.
5. Suggestion generation runs after the selected correction and should only
   expose viable alternatives.

## Structural pressure points

`correct_text_rich` currently contains several independent phrase dispatch
blocks. They overlap in responsibility and can consume the same input before a
more appropriate rule sees it:

- initial spaced preposition/article handling near `app.py:8005`;
- capitalised-place handling near `app.py:8530`;
- split preposition/article and negative handling near `app.py:9151`;
- `f`/`b`, `xi`/`bi`/`fi`, and `ta` handling near `app.py:9399`;
- compact and hyphenated article fallbacks near `app.py:9774`.

This is the source of many apparent "missing rule" bugs: a rule can exist but
never run because an earlier block has already emitted a phrase.

## Rules for the next changes

- Protected English and protected names never enter Maltese fuzzy correction.
- A preposition/article branch must first use a strict lexical tail. It may
  only use a fuzzy tail when the candidate still has the required part of
  speech.
- A phrase rule must not call a broad candidate search solely to decide whether
  it can consume its neighbour.
- Surface selection happens once, after a correction has been chosen. It must
  not be reimplemented inside phrase branches.
- Each regression needs an exact input/output test before adding a new path.
- If two historical tests express contradictory i-/j- surface rules, preserve
  the more recent language decision and update the older test deliberately.

## Verified English behaviour

- `car` remains English and offers `karozza` as its Maltese alternative.
- `for granted` remains one English phrase without quotation marks.
- `'Hi'` becomes `Hi`.
- `'aotomatic'` becomes `aotomatic`, without Maltese mutation, and is marked
  unrecognized.
- An unapproved quoted sentence is preserved without its quote marks and is
  marked unrecognized as one span.

## Existing failures to resolve in the phrase dispatcher

- `f sormok` previously became `F'isromok.` because the `f`/`b` branch accepted
  a distant imperative suffix candidate. The anchor guard now leaves it as
  `F sormok.`. Producing `f'sormok` requires lexical evidence that `sorm` is a
  noun; that base is not currently in the dictionaries, and it cannot be
  inferred safely from the surface alone.
- The existing `Wara jmorru` regression test expects `Wara imorru.`, but a
  later i-/j- rule specifies that an i-form is removed or changed to `j` after
  a preceding vowel. This needs one confirmed canonical expectation before the
  initial-vowel code is changed.
