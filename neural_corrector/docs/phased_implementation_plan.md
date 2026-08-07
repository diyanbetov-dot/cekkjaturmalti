# Phased Implementation Plan

## Phase 1: Data Foundation

Parse all manual pairs losslessly, infer analytical tags, report malformed and
suspicious examples, lock group-aware splits, and test reconstruction and
leakage boundaries.

## Phase 2: Custom Neural Baseline

Train the character edit tagger on approved training pairs plus traceable
training-only synthetic corruptions. Calibrate the edit threshold using only
validation data. Evaluate once on locked real, clean, and challenge sets.

## Phase 3: Error Analysis

Measure precision, recall, F-score, CER, WER, exact match, clean preservation,
overcorrection, undercorrection, latency, size, and qualitative failures.
Expand clean and challenge evaluation before increasing model complexity.

## Phase 4: Controlled Data Expansion

Add approved historical examples, then corpus-derived clean text and
suffix-derived synthetic examples. Keep source groups partitioned and rerun
the same locked evaluation.

## Phase 5: Architecture Experiments

Compare a character encoder-decoder and a BERTu encoder with a project-trained
correction head. Keep the custom model-only result as the control.

## Phase 6: Optional Evidence Ablations

Independently test BERTu context, corpus evidence, dictionary validation,
morphology features, suffix features, and narrow postprocessing. Retain only
components that improve held-out quality without materially increasing false
corrections, latency, memory, or model size.

## Phase 7: Product Adapter

Map structured neural edits to the existing suggestion UI, retain confidence
and alternatives, and deploy the winning configuration separately. Production
optimization and Cloud Run integration begin only after quality is established.

