# Mini LLM Final Automated Evaluation Report

Generation samples analyzed: **20**

## Model comparison

| Metric | 25k | 100k |
|---|---:|---:|
| Samples | 10 | 10 |
| Mean continuation words | 150.30 | 161.80 |
| Mean continuation characters | 725.8 | 790.2 |
| Mean sentence words (heuristic) | 12.87 | 9.89 |
| Distinct-1 | 0.5341 | 0.5134 |
| Distinct-2 | 0.8694 | 0.8707 |
| Distinct-3 | 0.9521 | 0.9523 |
| Repeated bigram occurrence ratio | 0.1306 | 0.1293 |
| Repeated trigram occurrence ratio | 0.0479 | 0.0477 |
| Repeated sentence count per sample | 0.000 | 0.200 |
| Terminal punctuation fraction (heuristic) | 0.700 | 0.600 |
| Near generation limit fraction (heuristic) | 0.300 | 0.400 |
| Near limit and no terminal punctuation (heuristic) | 0.300 | 0.400 |
| Best validation loss | 0.727712 | 0.686244 |
| Validation perplexity | 2.070339 | 1.986241 |

## Interpretation guidance

- Lower validation loss and perplexity indicate better next-token prediction on the same validation data; they do not guarantee better stories.
- Higher distinct-n indicates greater surface-level n-gram diversity. It does not measure coherence or meaning.
- Repetition ratios count repeated n-gram occurrences beyond their first occurrence; lower is less repetitive by this definition.
- Terminal punctuation is checked after stripping common closing quotation/bracket characters.
- The near-limit flag compares the UTF-8 byte length of the continuation with 90% of `max_new_tokens`. For a byte-level tokenizer this is a useful warning signal, but it is not an exact token count.
- The completion indicators are heuristics. Inspect individual rows in `final_evaluation_metrics.csv` for outliers.

## Limitations

- Only the saved prompts and samples are measured; this is not a comprehensive estimate of all possible generations.
- Distinct-n measures surface n-gram diversity, not semantic quality or factuality.
- Terminal punctuation and near-limit flags are heuristics, not proof of sentence completeness or truncation.
- The 25k and 100k runs used different training batch sizes, so dataset size is not isolated as the only changed variable.

## Files produced

- Per-sample diagnostics: `reports\evaluation\final_evaluation_metrics.csv`
- Machine-readable summary: `reports\evaluation\final_evaluation_summary.json`
- This report: `reports\evaluation\final_evaluation_report.md`
