# Experiment Record

## Purpose

Compare two trained versions of the Mini LLM decoder-only Transformer
using recorded validation metrics and automated fixed-prompt generation
diagnostics.

## Shared model configuration

  Setting                  Value
  ------------------------ --------------------------
  Architecture             Decoder-only Transformer
  Tokenizer                Byte-level
  Vocabulary size          259
  Context length           128
  Transformer layers       4
  Attention heads          4
  Embedding dimension      256
  Parameter count          3,258,624
  Training steps per run   10,000
  Learning rate            0.0003
  Weight decay             0.1
  Gradient clipping        1.0
  Seed                     42

These values reflect the recorded project configuration. Check
`configs/tiny.json` and each run's training metadata before claiming
that a future run used identical settings.

## Run A --- 25k training subset

  -----------------------------------------------------------------------------------------------------------
  Field                               Recorded value
  ----------------------------------- -----------------------------------------------------------------------
  Run name                            `tiny_25k_baseline_b288_v1`

  Training subset                     25k

  Batch size                          288

  Hardware                            NVIDIA GeForce RTX 3050 Laptop GPU

  Steps                               10,000

  Best step                           10,000

  Best validation loss                0.7277123663490389

  Best validation perplexity          2.0703390090356906

  Latest validation loss              0.7277123663490389

  Latest validation perplexity        2.0703390090356906

  Validation evaluations              21

  Elapsed time at latest evaluation   3,146.53 seconds (about 52.44 minutes)

  Peak GPU memory                     About 2.64 GB, from the recorded run notes

  Checkpoint                          `artifacts/experiments/tiny_25k_baseline_b288_v1/checkpoints/best.pt`
  -----------------------------------------------------------------------------------------------------------

## Run B --- 100k training subset

  ----------------------------------------------------------------------------------------------------
  Field                               Recorded value
  ----------------------------------- ----------------------------------------------------------------
  Run name                            `tiny_100k_b1280_v1`

  Training subset                     100k

  Batch size                          1,280

  Hardware                            Google Colab Tesla T4, 15 GB VRAM

  Steps                               10,000

  Best step                           10,000

  Best validation loss                0.6862436973681224

  Best validation perplexity          1.986240582216629

  Latest validation loss              0.6862436973681224

  Latest validation perplexity        1.986240582216629

  Validation evaluations              21

  Elapsed time at latest evaluation   12,197.83 seconds (about 203.30 minutes)

  Checkpoint                          `artifacts/experiments/tiny_100k_b1280_v1/checkpoints/best.pt`
  ----------------------------------------------------------------------------------------------------

## Validation comparison

The 100k run's recorded validation loss is lower than the 25k run's:

-   25k: 0.7277123663490389
-   100k: 0.6862436973681224

Perplexity is also lower:

-   25k: 2.0703390090356906
-   100k: 1.986240582216629

This supports better next-token prediction by the 100k run on the
recorded validation setup. It does not by itself establish superior
narrative quality.

## Generation evaluation setup

-   10 fixed prompts per model, for 20 saved generations total.
-   Same prompt set and sampling configuration for both models.
-   Seed: 42.
-   Temperature: 0.8.
-   Top-k: 20.
-   Top-p: 0.9.
-   Configured maximum generation length: 1,000 tokens.
-   Device in evaluation script: CUDA.
-   Canonical metrics analyze the continuation after removing the
    prompt, when the saved text starts with the prompt.

## Canonical generation diagnostics

  Metric                                              25k       100k
  -------------------------------------------- ---------- ----------
  Samples                                              10         10
  Mean continuation words                           150.3      161.8
  Mean continuation characters                      725.8      790.2
  Distinct-1                                     0.534121   0.513413
  Distinct-2                                     0.869351   0.870710
  Distinct-3                                     0.952075   0.952299
  Repeated bigram occurrence ratio               0.130649   0.129290
  Repeated trigram occurrence ratio              0.047925   0.047701
  Repeated sentence count per sample                  0.0        0.2
  Terminal punctuation fraction (heuristic)          0.70       0.60
  Near generation-limit fraction (heuristic)         0.30       0.40

Use `reports/evaluation/final_evaluation_summary.json` as the current
source of aggregate values. Re-run `scripts/finalize_evaluation.py`
after changing the generation CSV.

## Interpretation

The 100k run has better recorded validation loss and perplexity,
slightly higher Distinct-2/Distinct-3, and slightly lower repeated
bigram/trigram occurrence ratios. Its average continuation is longer.
The 25k run has a higher terminal-punctuation fraction and a lower
near-limit flag fraction. These results are mixed and should not be
reduced to a single overall quality score.

## Reproducibility artifacts

-   Run logs: `reports/experiments/<run_name>/training_metrics.jsonl`
-   Validation comparison:
    `reports/evaluation/validation_comparison.csv`
-   Generated samples: `reports/evaluation/generation_comparison.csv`
-   Per-sample diagnostics:
    `reports/evaluation/final_evaluation_metrics.csv`
-   Aggregate diagnostics:
    `reports/evaluation/final_evaluation_summary.json`
-   Canonical report: `reports/evaluation/final_evaluation_report.md`

The checkpoints and large data files are excluded from Git. Keep local
copies backed up separately if you need to reproduce generation without
retraining.
