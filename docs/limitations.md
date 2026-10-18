# Limitations and Validity Notes

## Model and tokenizer

-   The model has 3,258,624 parameters and is intentionally small
    compared with modern production language models.
-   The tokenizer is byte-level with a vocabulary size of 259. It avoids
    a separate learned vocabulary but tokenizes text at a low linguistic
    level and can spend multiple tokens on ordinary words.
-   The configured context length is 128 tokens. Long-range dependencies
    are therefore limited by the model's context window and generation
    setup.

## Training comparison

-   The 25k and 100k runs used different batch sizes: 288 and 1,280.
-   The runs used different hardware: an RTX 3050 Laptop GPU and a Colab
    Tesla T4.
-   Both runs completed 10,000 steps, but the difference in data subset
    is not the only experimental difference. The observed metric gap
    cannot be attributed solely to training-data size.
-   The current comparison is useful as an engineering experiment, not a
    fully controlled causal study.

## Validation metrics

-   Validation loss and perplexity measure next-token prediction on the
    validation distribution.
-   Lower validation loss does not guarantee better story coherence,
    factuality, instruction following, or user preference.
-   Validation comparisons are meaningful only if both runs use the same
    validation split, tokenizer, preprocessing, and evaluation
    implementation. Verify those details in the data manifest and run
    configuration before making stronger claims.

## Generation evaluation

-   The current evaluation uses 10 prompts per model. This is a small
    sample and may not represent performance across different prompt
    types.
-   Both models use fixed prompts and common decoding settings, which
    improves comparability, but one seed does not characterize the full
    variability of sampling.
-   Distinct-n metrics measure surface-level lexical diversity. They do
    not assess semantic coherence or whether the generated text makes
    sense.
-   Repeated n-gram and repeated-sentence metrics capture only the
    specific repetition patterns defined by the scripts.
-   Terminal punctuation is a heuristic. A final punctuation mark does
    not guarantee a complete or coherent sentence; a missing mark does
    not prove truncation.
-   The near-generation-limit indicator compares continuation byte
    length to the configured limit as a rough warning. It is not an
    exact count of tokens produced by the tokenizer.
-   Generation runtime includes process startup and checkpoint loading.
    It should not be reported as pure inference throughput.
-   No human evaluation is included in this project's current evaluation
    plan. Consequently, the report does not claim that automated metrics
    fully measure perceived writing quality.

## Data and reproducibility

-   The full corpus, training subsets, test data, and validation data
    are excluded from Git because of their size.
-   A fresh clone will not include these data files or model
    checkpoints. Follow the manifest and data-preparation code to
    reconstruct the local setup.
-   The generation evaluation requires the specified checkpoints at
    their expected local paths, or corresponding path updates in the
    evaluation script.
-   Python, PyTorch, CUDA, and driver versions can affect runtime and
    hardware behavior. Record exact versions when reproducing
    experiments.
-   Random seeds improve repeatability but do not guarantee
    bitwise-identical results across hardware, CUDA versions, or
    nondeterministic kernels.

## Appropriate claims

Supported by the current records: - The 100k run has lower recorded
validation loss and perplexity than the 25k run. - The automatic
generation diagnostics show small differences in surface-level
diversity, repetition, output length, and terminal punctuation. - The
project includes an automated evaluation pipeline and a test suite
reported to pass 85 tests.

Not established by the current records: - That increasing the dataset
from 25k to 100k alone caused the validation improvement. - That either
model is generally better at storytelling. - That automatic diversity or
punctuation metrics establish semantic quality. - That the measured
generation runtime is an inference-throughput benchmark.
