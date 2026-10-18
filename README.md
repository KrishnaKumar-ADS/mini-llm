# Mini LLM Lab

A small decoder-only Transformer language model built from scratch in
PyTorch. This project documents the data pipeline, training runs, text
generation, and automated comparison of two checkpoints trained on 25k
and 100k training examples.

> **Project status:** The two training runs and the automated evaluation
> pipeline are complete. The repository is being prepared for
> reproducibility and publication. Results below are the recorded
> results from the current experiments, not a claim that the model
> produces consistently coherent stories.

## Highlights

-   Decoder-only Transformer implemented in PyTorch.
-   Byte-level tokenizer with a vocabulary size of 259.
-   Context length: 128 tokens.
-   Model configuration: 4 Transformer layers, 4 attention heads,
    embedding dimension 256.
-   Parameter count: 3,258,624.
-   Two trained runs: 25k and 100k training subsets, each trained for
    10,000 steps.
-   Automated fixed-prompt generation comparison and text-statistics
    reports.
-   Automated test suite: 85 tests passed in the reported local run.

## Repository layout

``` text
mini-llm-lab/
├── configs/
│   └── tiny.json
├── data/
│   ├── dataset_manifest.json
│   └── ...                  # local datasets; intentionally excluded from Git
├── reports/
│   └── evaluation/
│       ├── generation_comparison.csv
│       ├── validation_comparison.csv
│       ├── final_evaluation_report.md
│       ├── final_evaluation_metrics.csv
│       └── final_evaluation_summary.json
├── scripts/
│   ├── train_scratch.py
│   ├── generate.py
│   ├── evaluate_generations.py
│   ├── analyze_evaluation.py
│   ├── extend_evaluation_metrics.py
│   └── finalize_evaluation.py
├── src/
├── tests/
├── .gitignore
├── pyproject.toml
└── README.md
```

The tree above highlights the main project areas. Other source modules
and tests may exist in the repository.

## Requirements

-   Python 3.11 is the version used in the recorded local Windows
    environment.
-   PyTorch.
-   Dependencies declared in `pyproject.toml`.
-   CUDA-capable NVIDIA GPU is recommended for training and GPU
    generation. CPU may be usable for some tasks but will be slower.

Use a virtual environment in the project root. In PowerShell:

``` powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the project using the packaging configuration in
`pyproject.toml`. If the project defines an installable package and its
dependencies there, use:

``` powershell
python -m pip install -e .
```

If that command fails, inspect `pyproject.toml` and follow its declared
dependency/install instructions rather than installing arbitrary package
versions.

## Data

The training corpus and split files are deliberately excluded from
version control because the local files are large.
`data/dataset_manifest.json` is retained as lightweight metadata.

Expected local files used by the project include:

-   `data/subsets/train_25k.txt`
-   `data/subsets/train_100k.txt`
-   `data/validation.txt`
-   `data/test.txt`

The project also has larger corpus and subset files. Consult
`data/dataset_manifest.json` and the data-preparation code for the
precise source, preprocessing rules, and split construction. Recreate or
obtain the data before running training or evaluation. Do not assume the
dataset is included in a GitHub clone.

## Training runs

Two checkpoints are compared:

  ---------------------------------------------------------------------------------------------------
  Run                             Training Batch size      Steps Hardware           Best   Perplexity
                                    subset                                    validation 
                                                                                    loss 
  ----------------------------- ---------- ---------- ---------- ---------- ------------ ------------
  `tiny_25k_baseline_b288_v1`          25k        288     10,000 NVIDIA RTX     0.727712     2.070339
                                                                 3050                    
                                                                 Laptop GPU              

  `tiny_100k_b1280_v1`                100k      1,280     10,000 Colab          0.686244     1.986241
                                                                 Tesla T4                
                                                                 (15 GB                  
                                                                 VRAM)                   
  ---------------------------------------------------------------------------------------------------

Both recorded runs reached their best and latest validation result at
step 10,000, with 21 validation evaluations each. The detailed run
information is in
[`docs/experiment-record.md`](docs/experiment-record.md).

**Important experimental limitation:** the batch sizes differ (288
vs. 1,280), and the runs used different hardware. This is an observed
comparison of these two runs, not a controlled experiment isolating
training-data size alone.

## Generate text

The generation script supports checkpoint selection, prompts, output
length, sampling parameters, seed, and device selection. Example
PowerShell command for the 25k model:

``` powershell
python scripts/generate.py `
  --checkpoint artifacts/experiments/tiny_25k_baseline_b288_v1/checkpoints/best.pt `
  --prompt "Once upon a time" `
  --max-new-tokens 200 `
  --temperature 0.8 `
  --top-k 20 `
  --top-p 0.9 `
  --seed 42 `
  --device auto
```

For the 100k model, replace the checkpoint path with:

``` text
artifacts/experiments/tiny_100k_b1280_v1/checkpoints/best.pt
```

Checkpoints are not included in Git. Restore or train the relevant
checkpoint before running generation.

## Run the evaluation

The evaluation compares both checkpoints over the same set of 10
prompts, using seed 42 and the same sampling settings. The generation
script is currently configured for `MAX_NEW_TOKENS = 1000`.

``` powershell
python scripts/evaluate_generations.py
python scripts/analyze_evaluation.py
python scripts/extend_evaluation_metrics.py
python scripts/finalize_evaluation.py
```

The canonical continuation-based report is:

-   `reports/evaluation/final_evaluation_report.md`

Its supporting outputs are:

-   `reports/evaluation/final_evaluation_metrics.csv` --- per-sample
    diagnostic metrics.
-   `reports/evaluation/final_evaluation_summary.json` ---
    machine-readable aggregate metrics.
-   `reports/evaluation/generation_comparison.csv` --- generated text
    for paired prompts.
-   `reports/evaluation/validation_comparison.csv` --- validation
    metrics for each run.

The older `evaluation_summary.md` and `extended_metrics_summary.md` are
legacy full-text analyses, calculated over the full saved text including
the prompt. They should not be compared directly with continuation-only
metrics without accounting for the different measurement scope.

## Current results

### Validation

  Metric                           25k model   100k model
  ------------------------------ ----------- ------------
  Best validation loss              0.727712     0.686244
  Perplexity at best-loss step      2.070339     1.986241
  Validation evaluations                  21           21

The 100k run has lower recorded validation loss and perplexity. These
metrics measure next-token prediction on the validation data; they do
not guarantee better narrative coherence.

### Generation diagnostics

Based on the current canonical report and 10 generated samples per
model:

  Metric                                         25k model   100k model
  -------------------------------------------- ----------- ------------
  Mean continuation words                            150.3        161.8
  Distinct-2                                        0.8694       0.8707
  Terminal punctuation fraction (heuristic)           0.70         0.60
  Near generation-limit fraction (heuristic)          0.30         0.40

These diagnostics are calculated over a small prompt set. Distinct-n
measures surface n-gram diversity, not meaning. Punctuation and
near-limit flags are heuristics, not proof of sentence completion or
truncation.

## Tests

Run the test suite from the repository root:

``` powershell
python -m pytest -q
```

The reported local test run passed all 85 tests. Re-run the tests in
your own environment after changing code or dependencies.

## Limitations

-   This is a small language model; generations can contain grammatical
    errors, repetition, semantic inconsistency, and incomplete
    sentences.
-   Byte-level tokenization makes the vocabulary compact but does not
    provide subword or word-level linguistic units.
-   Generation metrics are automatic surface-level diagnostics, not a
    complete measure of fluency, coherence, factuality, or usefulness.
-   The generation comparison uses 10 prompts per model; conclusions may
    not generalize to other prompts.
-   The 25k and 100k experiments use different batch sizes and hardware,
    so dataset size is not isolated as the only changing variable.
-   Generation runtime includes process startup and checkpoint loading;
    it is not a clean token-throughput benchmark.
-   Large datasets and checkpoints are excluded from Git; a fresh clone
    requires data preparation and access to checkpoints or a training
    run.

See [`docs/limitations.md`](docs/limitations.md) for the expanded
limitations record and
[`docs/experiment-record.md`](docs/experiment-record.md) for run
details.
