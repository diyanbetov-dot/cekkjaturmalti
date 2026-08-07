# Baseline Results

## Configuration

- Model: custom two-layer bidirectional GRU character edit tagger
- Trainable parameters: 481,068
- Checkpoint size: 1,930,296 bytes
- Training device: CPU (`torch 2.13.0+cpu`)
- Training examples after chunking and augmentation: 528
- Validation examples: 20
- Best validation loss: 1.4938 at epoch 10
- Full first-run training time: 229.1 seconds
- Selected action threshold: 0.85
- Threshold source: validation only
- Peak evaluation process working set: 227,872,768 bytes

All optional components were disabled: BERTu, corpus, dictionaries,
morphology, suffix runtime, old preprocessor candidates, and postprocessing.

## Locked Evaluation

| Split | Exact | Precision | Recall | F1 | CER | WER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 5.0% | 64.6% | 33.7% | 44.3% | 13.6% | 67.7% |
| Real test | 10.0% | 78.7% | 50.0% | 61.2% | 9.1% | 33.6% |
| Clean test | 57.1% preserved | n/a | n/a | n/a | 1.7% | 10.3% |
| Challenge | 0.0% | 71.1% | 30.8% | 42.9% | 8.7% | 47.2% |

Mean inference time was about 0.010 seconds per real-test example and 0.183
seconds per challenge example on the local CPU. These timings exclude process
startup and model loading.

## Interpretation

The experiment proves that the project data can train a small independent
neural corrector that learns useful Maltese transformations. For example, the
model produces `Mort il-baħar.` from `mort il bahar`, and the bare-word n-best
route produces `agħmel` with `għamel` as a second neural alternative for
`amel`.

It is not ready to replace the frozen hybrid checker. Exact sentence accuracy
is low and 42.9% of clean test sentences were changed unnecessarily. The next
quality investment should be substantially more clean-text preservation data
and more manually corrected contextual pairs, followed by another custom-model
run. Optional corpus, BERTu, dictionary, and morphology components should still
enter only as separately measured ablations.

